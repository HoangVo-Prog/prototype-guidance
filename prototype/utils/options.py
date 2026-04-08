import argparse

def get_prototype_cli(parser=None):
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--embedding_dim', type=int, default=512)

    return parser    
    