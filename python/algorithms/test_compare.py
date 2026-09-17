import unittest
from pyspla import FLOAT, INT, Matrix
import math
from collections import defaultdict
from bfs_classic import bfs_naive as bfs_n
from bfs_spla import bfs_spla as bfs_s
from pr_classic import pagerank_naive as pr_n
from pr_spla import pr_spla as pr_s
from sssp_classic import sssp_naive as sssp_n
from sssp_spla import sssp_spla as sssp_s
from tc_classic import tc_naive as tc_n
from tc_spla import tc_spla as tc_s

INF = math.inf
def vector_list(v, n):
    idx, vals = v.to_lists()
    depth = [None] * n
    for i, val in zip(idx, vals):
        depth[i] = val - 1  
    return depth
def build_graph(edges, n, directed=True, upper=False, weighted=False):
    if upper:
        edges = [(u, v) if u < v else (v, u) for u, v, *_ in edges]
    if not weighted:
        edges = [(u, v, 1) for u, v, *_ in edges]
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w) if weighted else v)
        if not directed:
            graph[v].append((u, w) if weighted else u)
    rows, cols, vals = zip(*sorted(edges)) if edges else ([], [], [])
    dtype = FLOAT if weighted else INT
    A = Matrix.from_lists(list(rows), list(cols), list(vals), (n, n), dtype)
    return graph, A

def build_pr(edges, n, alpha):
    graph, _ = build_graph(edges, n)
    Ai = [graph[i] for i in range(n)]
    Ax = [[alpha / len(graph[i]) for _ in graph[i]] for i in range(n)]
    rows, cols, vals = [], [], []
    for u in range(n):
        for v in graph[u]:
            rows.append(u)
            cols.append(v)
            vals.append(alpha / len(graph[u]))
    rows, cols, vals = zip(*sorted(zip(rows, cols, vals))) if rows else ([], [], [])
    A = Matrix.from_lists(list(rows), list(cols), list(vals), (n, n), FLOAT)
    return Ai, Ax, A
class TestCompareBfs(unittest.TestCase):
    def bfs_equal(self, n, edges, start):
        graph, A = build_graph(edges, n)
        depth_n = bfs_n(start, graph, n)
        v_s, count_s, depth_s = bfs_s(start, A)
        depth_s = vector_list(v_s, n)
        self.assertEqual(depth_n, depth_s)

    def test_single(self):
        self.bfs_equal(1, [], 0)

    def test_edge(self):
        self.bfs_equal(2, [(0,1)], 0)

    def test_line(self):
        self.bfs_equal(4, [(0,1), (1,2), (2,3)], 0)

    def test_star(self):
        self.bfs_equal(4, [(0,1), (0,2), (0,3)], 0)

    def test_tree(self):
        self.bfs_equal(6, [(0,1), (0,2), (1,3), (1,4), (2,5)], 0)

    def test_cycle(self):
        self.bfs_equal(4, [(0,1), (1,2), (2,3), (3,0)], 0)

    def test_diamond(self):
        self.bfs_equal(4, [(0,1), (0,2), (1,3), (2,3)], 0)

    def test_start_not_zero(self):
        self.bfs_equal(3, [(1,0), (1,2)], 1)

class TestCompareTc(unittest.TestCase):
    def tc_equal(self, n, edges):
        graph, A = build_graph(edges, n, directed=False, upper=True)
        self.assertEqual(tc_n(graph, n), tc_s(A))

    def test_no_edges(self):
        self.tc_equal(3, [])

    def test_one_edge(self):
        self.tc_equal(2, [(0, 1)])

    def test_two_edges(self):
        self.tc_equal(3, [(0, 1), (1, 2)])

    def test_triangle(self):
        self.tc_equal(3, [(0, 1), (1, 2), (0, 2)])

    def test_square(self):
        self.tc_equal(4, [(0, 1), (1, 2), (2, 3), (0, 3)])

    def test_k4(self):
        self.tc_equal(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    def test_two_triangles_share_edge(self):
        self.tc_equal(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    def test_two_triangles_share_vertex(self):
        self.tc_equal(5, [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)])

    def test_triangle_with_tail(self):
        self.tc_equal(4, [(0, 1), (0, 2), (1, 2), (2, 3)])

    def test_cycle5(self):
        self.tc_equal(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])

    def test_star(self):
        self.tc_equal(4, [(0, 1), (0, 2), (0, 3)])

    def test_k5(self):
        self.tc_equal(5, [(i, j) for i in range(5) for j in range(i + 1, 5)])
class TestCompareSssp(unittest.TestCase):
    def sssp_equal(self, n, edges, start):
        graph, A = build_graph(edges, n, weighted=True)
        dist_n = sssp_n(start, graph, n)
        v_s = sssp_s(start, A)
        idx, vals = v_s.to_lists()
        dist_s = [INF] * n
        for i, val in zip(idx, vals):
            dist_s[i] = val
        dist_s[start] = 0.0

        for i, (a, b) in enumerate(zip(dist_n, dist_s)):
            if math.isinf(a):
                self.assertTrue(math.isinf(b), f"v{i}: classic={a}, spla={b}")
            else:
                self.assertAlmostEqual(a, b, places=3,
                                       msg=f"v{i}: classic={a}, spla={b}")

    def test_one_edge(self):
        self.sssp_equal(2, [(0, 1, 2.7)], 0)

    def test_line(self):
        self.sssp_equal(4, [(0, 1, 1.3), (1, 2, 2.4), (2, 3, 3.1)], 0)

    def test_two_paths(self):
        self.sssp_equal(3, [(0, 1, 5.5), (1, 2, 2.2), (0, 2, 9.7)], 0)

    def test_diamond(self):
        self.sssp_equal(4, [(0, 1, 2.4), (0, 2, 1.1), (1, 3, 1.8), (2, 3, 5.3)], 0)

    def test_disconnected(self):
        self.sssp_equal(4, [(0, 1, 1.6), (2, 3, 4.2)], 0)

    def test_start_not_zero(self):
        self.sssp_equal(3, [(1, 0, 2.5), (1, 2, 3.7)], 1)

    def test_big_weights(self):
        self.sssp_equal(3, [(0, 1, 1000.5), (1, 2, 2000.4), (0, 2, 5000.9)], 0)

    def test_cycle(self):
        self.sssp_equal(3, [(0, 1, 1.1), (1, 2, 2.3), (2, 0, 0.7)], 0)
        
class TestComparePr(unittest.TestCase):
    def pr_equal(self, n, edges, alpha=0.85, eps=1e-6):
        Ai, Ax, A = build_pr(edges, n, alpha)
        p_n = pr_n(Ai, Ax, alpha, eps)
        v_s = pr_s(A, alpha, eps)
        idx, vals = v_s.to_lists()
        p_s = [0.0] * n
        for i, val in zip(idx, vals):
            p_s[i] = val

        for i, (a, b) in enumerate(zip(p_n, p_s)):
            self.assertAlmostEqual(a, b, places=3,
                                   msg=f"v{i}: classic={a}, spla={b}")

    def test_single(self):
        self.pr_equal(1, [])

    def test_one_edge(self):
        self.pr_equal(2, [(0, 1)])

    def test_line(self):
        self.pr_equal(4, [(0, 1), (1, 2), (2, 3)])

    def test_cycle(self):
        self.pr_equal(3, [(0, 1), (1, 2), (2, 0)])

    def test_star(self):
        self.pr_equal(4, [(0, 1), (0, 2), (0, 3)])

    def test_dangling(self):
        self.pr_equal(3, [(0, 1), (1, 2)])

    def test_k4(self):
        self.pr_equal(4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])

    def test_diamond(self):
        self.pr_equal(4, [(0,1), (0,2), (1,3), (2,3)])

if __name__ == "__main__":
    unittest.main()