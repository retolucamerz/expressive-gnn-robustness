zinc12k_node_attr = {0: [0, 2, 1, 4, 5, 3, 6, 8, 7, 11, 9, 12, 10, 13, 15, 14, 16, 20, 18, 19, 17]}
zinc12k_edge_attr = {0: [1, 2, 3]}

# fmt: off
## only needed for the gradient attacks
occuring_MolHIV_atom_vals = {0: [5, 7, 6, 15, 16, 8, 14, 34, 52, 13, 10, 33, 49, 28, 26, 4, 27, 25, 77, 45, 29, 32, 18, 31, 41, 24, 44, 76, 79, 23, 73, 82, 43, 78, 50, 2, 30, 0, 81, 74, 51, 12, 80, 11, 46, 19, 21, 91, 39, 22, 64, 88, 54, 66, 63],
 1: [0],
 2: [3, 4, 2, 1, 5, 0, 6, 9, 7, 8, 10],
 3: [5, 6, 4, 3, 1, 8, 2, 7, 0],
 4: [0, 1, 2, 3, 4],
 5: [0, 2],
 6: [1, 2, 0, 5, 3, 4],
 7: [0, 1],
 8: [1, 0]}
occuring_MolHIV_bond_vals = {0: [0, 1, 2, 3], 1: [0], 2: [1, 0]}
# fmt: on

# fmt: off
ogbmol_atom_attr = {
    0: [5, 7, 6, 15, 16, 8, 14, 34, 52, 13, 10, 33, 49, 28, 26, 4],
} | {i: lambda node_feat: [node_feat[i].item()] for i in range(1, 9)}
ogbmol_bond_attr = {0: [0, 3, 1, 2], 1: [0], 2: [1, 0]}
# fmt: on

imdb_node_attr = {0: [0]}
imdb_edge_attr = {0: [0]}

mutag_node_attr = {0: list(range(7))}
mutag_edge_attr = {0: list(range(4))}