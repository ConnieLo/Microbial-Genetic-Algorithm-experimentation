import numpy as np
from matplotlib import pyplot as plt
import copy

cities = np.array([(31, 254), (359, 211), (217, 397), (343, 355), (457, 36), (276, 307), (386, 360), (481, 49), (447, 179), (90, 20),(388, 0), (381, 439), 
                    (392, 213), (56, 71), (381, 331), (51, 363), (370, 314), (323, 73), (426, 222), (464, 144), (345, 457), (425, 137), (205, 437)])

pop_size = 100

tourney_size = 2

max_gens = 15000

const_deme = 20

max_deme = 20

deme_step = 4

const_p_crossover = 0.5

p_crossover_step = 1

class MicroGA:
  def __init__(self, pop_size: int, generations: int, cities, tourney_size, deme_size, p_crossover):
    self.cities = cities
    self.pop_size = pop_size
    self.generations = generations
    self.tourney_size = tourney_size
    self.num_items = len(self.cities)
    self.deme_size = deme_size
    self.p_crossover = p_crossover
    self.genotype_len = len(self.cities)
    # initialise the population
    self.pop = self.initialise_pop()
    self.tourney_track = np.zeros((generations), dtype=int)
    self.best_fit = self.find_best()

  def initialise_pop(self): #generates the initial starting population
    pop = np.zeros((self.pop_size, self.num_items), dtype=int)
    for i in range(pop.shape[0]):
      pop[i, 1:] = np.random.choice(range(1, self.num_items), self.num_items-1, replace=False) #generates all indices of the cities in random order (always starts with 0)
    return np.squeeze(pop)
  
  def find_best(self): #finds the fitness of the best genotype in entire population
    best_fit = 0
    for g in self.pop:
      fit = self.fitness_function(g)
      if fit > best_fit:
        best_fit = copy.deepcopy(fit)
    return best_fit
  
  def generation(self, gen):#defines full process of gene transferral
    w, l = self.tournament()
    lg = self.crossover(self.pop[w], self.pop[l])
    lg = self.mutate(lg)
    self.pop[l] = copy.deepcopy(lg) #replaces losing genotype with crossed over and mutated version
    lg_fit = self.fitness_function(lg)
    if self.best_fit < lg_fit: #if new genotype is better than the current best, it's fitness is stored in best_fit attribute
      self.best_fit = copy.deepcopy(lg_fit)
    self.tourney_track[gen] = copy.deepcopy(self.best_fit)
  
  def tournament(self):
    g1_ind = np.random.randint(self.pop_size)
    deme_pos = np.random.randint(self.deme_size) - (self.deme_size // 2) #generates index within region of other genotype
    g2_ind = g1_ind + deme_pos
    if g2_ind > 99: #loops back to start if index goes over end of population
      g2_ind -= 99
    g1_fit = self.fitness_function(self.pop[g1_ind]) #gathers fitnesses of selected genotypes
    g2_fit = self.fitness_function(self.pop[g2_ind])
    if g1_fit > g2_fit: #evaluates the winner
      w = g1_ind
      l = g2_ind
    else:
      w = g2_ind
      l = g1_ind

    return w, l
  
  def crossover(self, g1, g2):
    for i, g in enumerate(g1):
      if i>0: #doesn't operate on first gene as this is the starting city
        p = np.random.randint(0, 10)
        if p <= self.p_crossover*10: #can crossover on some probability
          old_ind = np.squeeze(np.where(g2 == g)) #finds the current index of this city *g* in g2
          g2 = np.delete(g2, old_ind)
          g2 = np.insert(g2, i, g) #moves city in g2 to the same index as it is in g1
    return g2
  
  def mutate(self, g):
    num1 = np.random.randint(1, g.size)
    num2 = np.random.randint(1, g.size)
    while num1 == num2:
      num2 = np.random.randint(1, g.shape[0]) #makes sure that 2 selected indices are unique
    gene = copy.deepcopy(g[num1])
    g = np.delete(g, num1)
    g = np.insert(g, num2, gene) #moves one city in the genotype
    return g
  
  def fitness_function(self, genotype): #calculates fitness based on minimum distance
    total_distance = 0
    for i in range(1, genotype.size):
      coords = (self.cities[genotype[i-1]], self.cities[genotype[i]])
      total_distance += self.calc_distance(coords) #adds distance from previous city to the current to total
    coords = (self.cities[genotype[self.num_items-1]], self.cities[genotype[0]])
    total_distance += self.calc_distance(coords)
    fitness = (1000000/total_distance) #reciprocal of distance so that fitness increases as distance decreases. Multiplied by 1000000 to scale the difference
    return fitness
  
  def calc_distance(self, coords):
    x_2 = (coords[0][0]-coords[1][0])**2  #calculates the square of the difference in x coords
    y_2 = (coords[0][1]-coords[1][1])**2  #calculates the square of the difference in y coords
    c_2 = x_2 + y_2 #adds together
    return np.sqrt(c_2) #returns square root aka pythagorean distance
  
  def evolve(self):
    for i in range(self.generations): #runs as many generations as specified by generations attribute
      self.generation(i)

    return self.best_fit #returns the best fitness in the population after running

ga_dicts = {}
for i in range(10): #generates 10 dictionaries
  ga_dicts["ga_dict"+str(i)] = {} #generates 10 gas for each dictionary
  for j in range(1, 11, 1):
    ga_dicts["ga_dict"+str(i)]["ga_"+str(j)]= MicroGA(pop_size, max_gens, cities, tourney_size, j*10, 1) #note: this is currently testing Deme Size. Previously this loop was used to test crossover probability as well.

results_dict = {}

count = 0
for ga_dict in ga_dicts: #concatenates the tourney_track attribute from every GA in the dictionary
  results = []
  for ga in ga_dicts[ga_dict]:
    ga_dicts[ga_dict][ga].evolve()
    results.append(ga_dicts[ga_dict][ga].tourney_track)
    print("done")
  results_dict["res"+str(count)] = np.array(results)
  count+=1

full_results = np.zeros_like(results_dict["res0"])

for r in results_dict:
  full_results += results_dict[r] #adds the results of each ga to the results of the GAs with the same parameters

full_results = full_results/10 #gets the mean best fitness at every stage between every GA with the same parameters
 
print(full_results.shape[0])

for n, i in enumerate(full_results): #plots results
  plt.plot(range(0, i.size), i, label='Deme Size: '+str((n+1)*10))#note: Label reflects the attribute we were testing.
plt.legend(loc="lower right")
plt.xlabel('Tournaments')
plt.ylabel('Best Fitness')
plt.show()

