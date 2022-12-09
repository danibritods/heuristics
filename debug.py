import tsplib95 
import networkx as nx
import numpy as np
np.random.seed(42)

problem = tsplib95.load("ca4663.tsp")
nodes = problem.node_coords

class TSPSolver:
    def solve(self,nodes):
        self.nodes = nodes
        initial_solution = self.initial_solution()
        best_solution = self.local_search(initial_solution)

    def initial_solution(self):
        """path is the node_codes in order, cost is the from previous to current node."""
        n = len(self.nodes)
        # path = np.random.permutation(np.arange(1,n+1,1)) 

        path = list(self.nodes.keys())
        cost = self.cost(path)
        solution = {"path":path} | cost
        return solution

    def local_search(self, current_solution):
        self.simulated_annealing(current_solution)

    def simulated_annealing(self, current_solution):
        current_path = current_solution["path"]
        current_eval = current_solution["total_cost"]
        initial_eval = current_eval

        n = len(current_path)
        i,j = np.random.choice(range(1,n-1),size=2,replace=False)
        bounds = (i,j) if i<j else (j,i)
        swap_cost = self.swap_cost(current_path,bounds)

        if swap_cost < 0:
            current_path = self.swap_n_reverse(current_path, (i,j))
            current_eval -= 10 #swap_cost
            print(self.check_eval(current_eval,current_path))

        print(f"initial cost:{initial_eval}; current cost: {current_eval}")

    def node_distance(self,a,b):
        try: 
            distance = self.distance(self.nodes[a], self.nodes[b])
        except TypeError:
            print(f"TypeError in node_distance: a,b ={a,b}")
        return distance

    def distance(self, a, b):
        #todo: check performance
        return np.sqrt(
        (a[0] - b[0])**2 + (a[1] - b[1])**2
        )

        
    def swap_n_reverse(self, path, bounds):
        i,j = bounds
        path[i:j] = path[i:j][::-1]

        return path


    def check_eval(self, current_eval,current_path):
        calculated_cost = self.cost(current_path)["total_cost"]
        print(f"eval: {current_eval}; calculated: {calculated_cost}; diff: {calculated_cost-current_eval}")

        return ((calculated_cost - current_eval)**2)**0.5 < 100


    def swap_cost(self, path, bounds):
        i,j = bounds
        n1,n2 = path[i-1], path[i]
        n3,n4 = path[j], path[j+1]

        d = self.node_distance
        cost = -(d(n1,n2) - d(n1,n4)) + (d(n3,n4) - d(n2,n4))

        curr_cost = self.node_distance(path[i-1],path[i]) + self.node_distance(path[j],path[j+1])
        new_cost = self.node_distance(path[i-1],path[j]) + self.node_distance(path[i],path[j+1])
        swap_cost = new_cost - curr_cost
        print(f"old: {swap_cost} cost: {cost}")
        return cost
            
  
    def cost(self, path):
        n = len(path)
        cost = np.array([self.distance(self.nodes[path[i]],self.nodes[path[i+1]]) for i in range(n-1)])
        total_cost = sum(cost)
        # cost = np.zeros(n)
        # list(map(self.distance, path))
        # cost = [ self.di
        # stance(a,b) for a,b in path]
        return {"cost":cost, "total_cost":total_cost}
        

dic = {k:v for k,v in list(nodes.items())[:10]}
solver = TSPSolver()
# solver.swap_n_reverse(a)
# solver.solve(dic)
# solver.solve(nodes)

# solver.swap_n_reverse(list(dic.keys()),(1,8))
# solver.nodes = dic
# solver.swap_cost(list(dic.keys()),[2,3])
solver.solve(dic)
