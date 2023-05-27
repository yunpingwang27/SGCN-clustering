"""SGCN runner."""

from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs,get_true_comm,test_read_graph
from k_mean import cluster
# args = parameter_parser()
# edge = read_graph(args)
# print(edge['m_pos'])

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    # test_edges = test_read_graph()
    # comm = get_true_comm('./input/create/node_comm1.csv')
    # comm = get_true_comm('./output/spectral/test_spectral_mona.csv')
    # comm = get_true_comm(args)
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)
    cluster(args)
if __name__ == "__main__":
    main()
