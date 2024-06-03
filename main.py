import os
import sys
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List
from server import Server
from client import Client
from utils.config import Config
from torchvision.transforms import transforms
from utils.data import load_data, swap_proto_data
from utils.torchmodel import create_model
import pandas as pd

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
# os.environ["XLA_FLAGS"] = \
#     "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREAD"] = "1"
distill_threshold_acc = 0.3
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
# if len(sys.argv) > 2:
#     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    args = Config(cfg_path=sys.argv[1])
    df_save_file = sys.argv[1].replace("configs", "logs").replace(".yaml", ".csv")
    args.save_dir.client = os.path.expanduser(args.save_dir.client)
    args.save_dir.proto_data = os.path.expanduser(args.save_dir.proto_data)
    args.dataset.dir = os.path.expanduser(args.dataset.dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(precision=4)
    np.set_printoptions(precision=4, linewidth=200)

    # ============================Setup============================
    logger.info(f"Comm-Acc-Rec will be saved to {df_save_file}")
    logger.info(f"Loading dataset {args.dataset.name} from {args.dataset.dir}...")
    comm_acc_rec = {'acc': [], 'comm': []}
    dataset = load_data(args.dataset.name, args.dataset.dir, args.dataset.image_size, args.dataset.resize_extend)
    if len(dataset.train_data.data.shape) > 3:
        dataset.transpose_data()
        dataset.scale_by(1.0 / 255)
    train_transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(15),
    ])
    train_transform2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    train_transform = None 
    client_datasets = \
            dataset.get_heterogeneous_datasets(args.num_clients,
                                               args.dataset.num_classes,
                                               args.dataset.alpha,
                                               transform=train_transform)

    logger.info("Creating server...")
    server = Server(args.dataset.num_classes, args)
    clients: List[Client] = []
    model_types = args.model_types

    # ---------------------Dataset initialization---------------------
    # Client datasets preparation
    logger.info(f"Creating {args.num_clients} clients...")
    # Test dataset preparation
    mini_test_loader, full_test_loader = dataset.prepare_test_loaders(args.mini_test_proportion,
                                                                      args.test_batch_size,
                                                                      args.test_num_workers)

    for client_id, client_dataset in zip(range(args.num_clients), client_datasets):
        model_type = model_types[client_id % len(model_types)]
        # print(client_dataset.get_distribution(args.dataset.num_classes))
        model = create_model(model_type,
                             args.dataset.image_channels,
                             args.dataset.num_classes)
        dataset_args = {
            "batch_size": args.batch_size,
            "num_workers": 4,
        }
        optimizer_args = {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        client = Client(client_id=client_id,
                        model_type=model_type,
                        model=model,
                        dataset=client_dataset,
                        output_dim=args.dataset.num_classes,
                        dataset_args=dataset_args,
                        optimizer_args=optimizer_args,
                        num_prototypes_per_class=args.dd.num_prototypes_per_class,
                        criterion=args.client_loss,
                        device=args.device,
                        max_sync_data_num=2000)
        if args.resume:
            client.load_model(args.save_dir.client, str(client_id) + "-", "8")
        client.init_synthesized_data(resume=args.resume,
                                     save_dir=args.save_dir.proto_data,
                                     suffix=args.train_phases)
        clients.append(client)
        distribution = client.dataset.get_distribution(args.dataset.num_classes)
        client.personalized_test_loader = \
            dataset.get_personalized_test_loader(distribution,
                                                 test_transform=None,
                                                 collate_fn=None,
                                                 args=args)
    tqdm_ = partial(tqdm, ncols=100, leave=False)
    args.train_phases = 1
    # args.fed_phases = 1
    # clients = clients[:10]
    # ============================Training============================
    if not args.resume:
        # initialize_clients(clients)
        # ------------------------Isolated training-----------------------
        logger.info("Start isolated training")
        mini_test_rec_iter = 0
        for phase in range(args.train_phases):
            print(f"Phase {phase}/{args.train_phases} in isolated training")
            for i, client in tqdm(enumerate(clients), desc="Clients training...", ncols=100):
                for epoch in range(args.epoch_per_phase):
                    client.train(save_feat=(epoch == args.epoch_per_phase - 1))
                client.merge_feat()

                # logger.debug(f"Client {i} training finished.")
                client.eval_acc = client.eval(mini_test_loader)
                client.personalized_eval_acc = client.eval(client.personalized_test_loader)
                if client.personalized_eval_acc > distill_threshold_acc:
                    client.init_proto_optimizer()
                    for epoch in range(args.dd.epoch_per_phase):
                        client.modify_synthesized_data()
                client.clear_feat()
    else:
        for client in tqdm(clients, desc="Loading model...", ncols=100, leave=False):
            client.eval_acc = client.eval(mini_test_loader)
            client.personalized_eval_acc = client.eval(client.personalized_test_loader)
    tau = args.dataset.tau
    total_transmit = 0
    client_accs_detailed = np.zeros(args.num_clients, dtype=np.float32)
    for phase in range(args.fed_phases):
        st_time = time.time()
        logger.info(f"Start federated phase {phase}/{args.fed_phases}")
        print(f"Federated Phase: {phase}/{args.fed_phases}")
        swap_clients = []
        for client in clients:
            if client.personalized_eval_acc > distill_threshold_acc:
                x_proto, y_proto = client.get_synthesized_data()
                y_proto = torch.argmax(y_proto, dim=-1)
                x_proto = x_proto.cpu()
                y_proto = y_proto.cpu()
                server.save_to_knowledge_base(x_proto, y_proto)
                swap_clients.append(client)
        logger.info(f"Fed Phase{phase}/{args.fed_phases} - Swaping data...")
        # swap_proto_data(swap_clients, 4 * len(swap_clients))
        total_transmit += server.merge_new_data(replace=True, iteration=phase, save_dir=args.save_dir.proto_data)
        for client in tqdm(clients, desc="Synthesized data training", ncols=100):
            distribution = client.dataset.get_distribution(args.dataset.num_classes)
            new_X, new_y, client.transmit_size = server.get_distilled_data_by_distribution(distribution, tau, -1)
            client.save_synthesized_data(new_X, new_y, replace=True)

            # training
            modi_data = client.personalized_eval_acc > distill_threshold_acc
            for epoch in range(args.epoch_per_phase):
                save_feat = (epoch >= args.epoch_per_phase - 3) and modi_data
                step_lr = (epoch == args.epoch_per_phase - 1)
                client.train(save_feat=save_feat, step_lr=step_lr)
            # distill data
            if modi_data:
                client.merge_feat()
                client.init_proto_optimizer()
                for _ in range(args.dd.epoch_per_phase):
                    client.modify_synthesized_data()
                client.clear_feat()
                
            # test
            client.eval_acc = client.eval(mini_test_loader) * 100
            client.personalized_eval_acc = client.eval(client.personalized_test_loader) * 100
            client.train_acc = client.eval(client.data_loader) * 100
            total_transmit += client.transmit_size
            client_accs_detailed[client.client_id] = client.personalized_eval_acc
            comm_acc_rec['acc'].append(client_accs_detailed.mean())
            comm_acc_rec['comm'].append(total_transmit)

        client_accs = np.array([client.eval_acc for client in clients])
        personalized_client_accs = np.array([client.personalized_eval_acc for client in clients])
        client_train_accs = np.array([client.train_acc for client in clients])
        avg_acc = np.mean(client_accs)
        avg_personalized_acc = np.mean(personalized_client_accs)
        avg_train_acc = np.mean(client_train_accs)
        # print(f"[Federate] Test acc at phase {phase}: {client_accs}")
        print(f"[Federate] Mean test acc at phase {phase}: {avg_acc}")
        # print(f"[Federate] Personalized test acc at phase {phase}: {personalized_client_accs}")
        print(f"[Federate] Mean personalized test acc at phase {phase}: {avg_personalized_acc}")
        print(f"[Federate] Train acc at phase {phase}: {client_train_accs}")
        print(f"[Federate] Mean train acc at phase {phase}: {avg_train_acc}")
        print(f"[Federate] Fetch {np.mean(server.fetch_hist)} data per client")
        print(f"[Federate] Client transmit number: {server.fetch_hist}")
        print(f"[Federate] Transmit {total_transmit} data")
        print(f"[Federate] Client transmit size: {[client.transmit_size for client in clients]}")
        server.fetch_hist = []
    comm_acc_rec_df = pd.DataFrame.from_dict(comm_acc_rec)
    comm_acc_rec_df.to_csv(df_save_file)
