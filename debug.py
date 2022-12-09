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
        cooling_rate = 100
        non_improov = 0
        max_non_improov = 10

        temperature = 6000
        n = len(current_path)

        while non_improov < max_non_improov and temperature > 0:
            i,j = np.random.choice(range(1,n-1),size=2,replace=False)
            bounds = (i,j) if i<j else (j,i)
            swap_cost = self.swap_cost(current_path,bounds)

            if swap_cost < 0:
                current_path = self.swap_n_reverse(current_path, (i,j))
                current_eval += swap_cost
            else:
                non_improov += 1
                accepting_prob = np.e**(swap_cost/temperature)
                if np.random.uniform() < accepting_prob:
                    current_path = self.swap_n_reverse(current_path, (i,j))
                    current_eval += swap_cost
                
            temperature -= cooling_rate

    
        print(self.check_eval(current_eval,current_path, swap_cost))
        print(f"initial cost:{initial_eval}; current cost: {current_eval}; {current_eval/initial_eval}% \n ({i,j})")

    def node_distance(self,a,b):
        distance = self.distance(self.nodes[a], self.nodes[b])
        return distance

    def distance(self, a, b):
        #todo: check performance
        return np.sqrt(
        (a[0] - b[0])**2 + (a[1] - b[1])**2
        )

        
    def swap_n_reverse(self, path, bounds):
        new_path = np.copy(path)
        i,j = bounds
        new_path[i:j] = new_path[i:j][::-1]

        return new_path


    def check_eval(self, current_eval,current_path, swap_cost):
        calculated_cost = self.cost(current_path)["total_cost"]
        diff = calculated_cost - current_eval + 0.0000001
        print(f"\n swap/diff: {swap_cost/diff} \n eval: {current_eval}; calculated: {calculated_cost}; diff: {diff} diff% {current_eval/calculated_cost}")

        return ((calculated_cost - current_eval)**2)**0.5 < 100


    def swap_cost(self, path, bounds):
        i,j = bounds
        n1,n2 = path[i-1], path[i]
        n3,n4 = path[j-1], path[j]

        d = self.node_distance
        cost = (d(n1,n2) - d(n1,n3)) + (d(n3,n4) - d(n2,n4))
        test_path = np.copy(path[i-1:j+1])
        calc_cost_before = self.cost(test_path)["total_cost"]
        test_path[1:-1] = test_path[1:-1][::-1]
        calc_cost_after = self.cost(test_path)["total_cost"]
        calc_cost = calc_cost_before - calc_cost_after

        # curr_cost = d(path[i-1],path[i]) + d(path[j],path[j+1])
        # new_cost = d(path[i-1],path[j]) + d(path[i],path[j+1])
        # swap_cost = new_cost - curr_cost

        print(f"swap_cost: {cost}; cost_calc: {calc_cost}; %{cost/calc_cost}")

        return calc_cost
            
  
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
