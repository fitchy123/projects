############
############ ALTHOUGH I GIVE YOU THE 'BARE BONES' OF THIS PROGRAM WITH THE NAME
############ 'skeleton.py', YOU CAN RENAME IT TO ANYTHING YOU LIKE. HOWEVER, FOR
############ THE PURPOSES OF THE EXPLANATION IN THESE COMMENTS, I ASSUME THAT
############ THIS PROGRAM IS STILL CALLED 'skeleton.py'.
############
############ IF YOU WISH TO IMPORT STANDARD MODULES, YOU CAN ADD THEM AFTER THOSE BELOW.
############ NOTE THAT YOU ARE NOT ALLOWED TO IMPORT ANY NON-STANDARD MODULES!
############

import os
import sys
import time
import random

start_time = time.time()

############
############ NOW PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS.
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

def read_file_into_string(input_file, ord_range):
    the_file = open(input_file, 'r') 
    current_char = the_file.read(1) 
    file_string = ""
    length = len(ord_range)
    while current_char != "":
        i = 0
        while i < length:
            if ord(current_char) >= ord_range[i][0] and ord(current_char) <= ord_range[i][1]:
                file_string = file_string + current_char
                i = length
            else:
                i = i + 1
        current_char = the_file.read(1)
    the_file.close()
    return file_string

def remove_all_spaces(the_string):
    length = len(the_string)
    new_string = ""
    for i in range(length):
        if the_string[i] != " ":
            new_string = new_string + the_string[i]
    return new_string

def integerize(the_string):
    length = len(the_string)
    stripped_string = "0"
    for i in range(0, length):
        if ord(the_string[i]) >= 48 and ord(the_string[i]) <= 57:
            stripped_string = stripped_string + the_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def convert_to_list_of_int(the_string):
    list_of_integers = []
    location = 0
    finished = False
    while finished == False:
        found_comma = the_string.find(',', location)
        if found_comma == -1:
            finished = True
        else:
            list_of_integers.append(integerize(the_string[location:found_comma]))
            location = found_comma + 1
            if the_string[location:location + 5] == "NOTE=":
                finished = True
    return list_of_integers

def build_distance_matrix(num_cities, distances, city_format):
    dist_matrix = []
    i = 0
    if city_format == "full":
        for j in range(num_cities):
            row = []
            for k in range(0, num_cities):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    elif city_format == "upper_tri":
        for j in range(0, num_cities):
            row = []
            for k in range(j):
                row.append(0)
            for k in range(num_cities - j):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    else:
        for j in range(0, num_cities):
            row = []
            for k in range(j + 1):
                row.append(0)
            for k in range(0, num_cities - (j + 1)):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    if city_format == "upper_tri" or city_format == "strict_upper_tri":
        for i in range(0, num_cities):
            for j in range(0, num_cities):
                if i > j:
                    dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

def read_in_algorithm_codes_and_tariffs(alg_codes_file):
    flag = "good"
    code_dictionary = {}   
    tariff_dictionary = {}  
    if not os.path.exists(alg_codes_file):
        flag = "not_exist"  
        return code_dictionary, tariff_dictionary, flag
    ord_range = [[32, 126]]
    file_string = read_file_into_string(alg_codes_file, ord_range)  
    location = 0
    EOF = False
    list_of_items = []  
    while EOF == False: 
        found_comma = file_string.find(",", location)
        if found_comma == -1:
            EOF = True
            sandwich = file_string[location:]
        else:
            sandwich = file_string[location:found_comma]
            location = found_comma + 1
        list_of_items.append(sandwich)
    third_length = int(len(list_of_items)/3)
    for i in range(third_length):
        code_dictionary[list_of_items[3 * i]] = list_of_items[3 * i + 1]
        tariff_dictionary[list_of_items[3 * i]] = int(list_of_items[3 * i + 2])
    return code_dictionary, tariff_dictionary, flag

############
############ THE RESERVED VARIABLE 'input_file' IS THE CITY FILE UNDER CONSIDERATION.
############
############ IT CAN BE SUPPLIED BY SETTING THE VARIABLE BELOW OR VIA A COMMAND-LINE
############ EXECUTION OF THE FORM 'python skeleton.py city_file.txt'. WHEN SUPPLYING
############ THE CITY FILE VIA A COMMAND-LINE EXECUTION, ANY ASSIGNMENT OF THE VARIABLE
############ 'input_file' IN THE LINE BELOW iS SUPPRESSED.
############
############ IT IS ASSUMED THAT THIS PROGRAM 'skeleton.py' SITS IN A FOLDER THE NAME OF
############ WHICH IS YOUR USER-NAME, E.G., 'abcd12', WHICH IN TURN SITS IN ANOTHER
############ FOLDER. IN THIS OTHER FOLDER IS THE FOLDER 'city-files' AND NO MATTER HOW
############ THE NAME OF THE CITY FILE IS SUPPLIED TO THIS PROGRAM, IT IS ASSUMED THAT 
############ THE CITY FILE IS IN THE FOLDER 'city-files'.
############

input_file = "AISearchfile012.txt"

############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS.
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

if len(sys.argv) > 1:
    input_file = sys.argv[1]

the_particular_city_file_folder = "city-files"
    
if os.path.isfile("../" + the_particular_city_file_folder + "/" + input_file):
    ord_range = [[32, 126]]
    file_string = read_file_into_string("../" + the_particular_city_file_folder + "/" + input_file, ord_range)
    file_string = remove_all_spaces(file_string)
    print("I have found and read the input file " + input_file + ":")
else:
    print("*** error: The city file " + input_file + " does not exist in the folder '" + the_particular_city_file_folder + "'.")
    sys.exit()

location = file_string.find("SIZE=")
if location == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
comma = file_string.find(",", location)
if comma == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
num_cities_as_string = file_string[location + 5:comma]
num_cities = integerize(num_cities_as_string)
print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))

comma = comma + 1
stripped_file_string = file_string[comma:]
distances = convert_to_list_of_int(stripped_file_string)

counted_distances = len(distances)
if counted_distances == num_cities * num_cities:
    city_format = "full"
elif counted_distances == (num_cities * (num_cities + 1))/2:
    city_format = "upper_tri"
elif counted_distances == (num_cities * (num_cities - 1))/2:
    city_format = "strict_upper_tri"
else:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()

dist_matrix = build_distance_matrix(num_cities, distances, city_format)
print("   the distance matrix 'dist_matrix' has been built.")

############
############ YOU NOW HAVE THE NUMBER OF CITIES STORED IN THE INTEGER VARIABLE 'num_cities'
############ AND THE TWO_DIMENSIONAL MATRIX 'dist_matrix' HOLDS THE INTEGER CITY-TO-CITY 
############ DISTANCES SO THAT 'dist_matrix[i][j]' IS THE DISTANCE FROM CITY 'i' TO CITY 'j'.
############ BOTH 'num_cities' AND 'dist_matrix' ARE RESERVED VARIABLES AND SHOULD FEED
############ INTO YOUR IMPLEMENTATIONS.
############

############
############ THERE NOW FOLLOWS CODE THAT READS THE ALGORITHM CODES AND TARIFFS FROM
############ THE TEXT-FILE 'alg_codes_and_tariffs.txt' INTO THE RESERVED DICTIONARIES
############ 'code_dictionary' AND 'tariff_dictionary'. DO NOT AMEND THIS CODE!
############ THE TEXT FILE 'alg_codes_and_tariffs.txt' SHOULD BE IN THE SAME FOLDER AS
############ THE FOLDER 'city-files' AND THE FOLDER WHOSE NAME IS YOUR USER-NAME, E.G., 'abcd12'.
############

code_dictionary, tariff_dictionary, flag = read_in_algorithm_codes_and_tariffs("../alg_codes_and_tariffs.txt")

if flag != "good":
    print("*** error: The text file 'alg_codes_and_tariffs.txt' does not exist.")
    sys.exit()

print("The codes and tariffs have been read from 'alg_codes_and_tariffs.txt':")

############
############ YOU NOW NEED TO SUPPLY SOME PARAMETERS.
############
############ THE RESERVED STRING VARIABLE 'my_user_name' SHOULD BE SET AT YOUR USER-NAME, E.G., "abcd12"
############

my_user_name = "mccb22"

############
############ YOU CAN SUPPLY, IF YOU WANT, YOUR FULL NAME. THIS IS NOT USED AT ALL BUT SERVES AS
############ AN EXTRA CHECK THAT THIS FILE BELONGS TO YOU. IF YOU DO NOT WANT TO SUPPLY YOUR
############ NAME THEN EITHER SET THE STRING VARIABLES 'my_first_name' AND 'my_last_name' AT 
############ SOMETHING LIKE "Mickey" AND "Mouse" OR AS THE EMPTY STRING (AS THEY ARE NOW;
############ BUT PLEASE ENSURE THAT THE RESERVED VARIABLES 'my_first_name' AND 'my_last_name'
############ ARE SET AT SOMETHING).
############

my_first_name = "Joshua"
my_last_name = "Fitch"

############
############ YOU NEED TO SUPPLY THE ALGORITHM CODE IN THE RESERVED STRING VARIABLE 'algorithm_code'
############ FOR THE ALGORITHM YOU ARE IMPLEMENTING. IT NEEDS TO BE A LEGAL CODE FROM THE TEXT-FILE
############ 'alg_codes_and_tariffs.txt' (READ THIS FILE TO SEE THE CODES).
############

algorithm_code = "AC"

############
############ DO NOT TOUCH OR ALTER THE CODE BELOW! YOU HAVE BEEN WARNED!
############

if not algorithm_code in code_dictionary:
    print("*** error: the agorithm code " + algorithm_code + " is illegal")
    sys.exit()
print("   your algorithm code is legal and is " + algorithm_code + " -" + code_dictionary[algorithm_code] + ".")

############
############ YOU CAN ADD A NOTE THAT WILL BE ADDED AT THE END OF THE RESULTING TOUR FILE IF YOU LIKE,
############ E.G., "in my basic greedy search, I broke ties by always visiting the first 
############ city found" BY USING THE RESERVED STRING VARIABLE 'added_note' OR LEAVE IT EMPTY
############ IF YOU WISH. THIS HAS NO EFFECT ON MARKS BUT HELPS YOU TO REMEMBER THINGS ABOUT
############ YOUR TOUR THAT YOU MIGHT BE INTERESTED IN LATER.
############

added_note = ""

############
############ NOW YOUR CODE SHOULD BEGIN.
############

#parameters
max_iterations = num_cities*3
number_ants = 25
alpha = 1.0
beta = 4.0
p = 0.1
w = 6 #number of elite ants
length_candidate_lists = 8
two_or_three_opt = 2
number_ants_locally_searched = 25
#initial deposit is calculated further down and is number_ants/length NN tour

def two_opt(NN_tour, NN_tour_distance, node_nearest_neighbours):

    nflag = True
    while nflag == True:
        nflag = False
        i = 1
        while i < num_cities and nflag == False:
            j = i + 1
            while j <= num_cities and nflag == False:
                if (NN_tour[j - 1] not in node_nearest_neighbours[NN_tour[i - 1]]) and (NN_tour[i - 1] not in node_nearest_neighbours[NN_tour[j - 1]]):
                    j += 1
                    continue
                if j == num_cities:
                    max_cur = max(dist_matrix[NN_tour[i - 1]][NN_tour[i]], dist_matrix[NN_tour[j - 1]][NN_tour[0]])
                    min_new = min(dist_matrix[NN_tour[i - 1]][NN_tour[i]], dist_matrix[NN_tour[j - 1]][NN_tour[0]])
                else: 
                    max_cur = max(dist_matrix[NN_tour[i - 1]][NN_tour[i]], dist_matrix[NN_tour[j - 1]][NN_tour[j]])
                    min_new = min(dist_matrix[NN_tour[i - 1]][NN_tour[j - 1]], dist_matrix[NN_tour[i]][NN_tour[j]])
                if min_new > max_cur:
                    j += 1
                    continue
                new_tour1 = NN_tour[0:i] 
                new_tour2 = list(reversed(NN_tour[i: j]))
                new_tour3 = NN_tour[j: num_cities]
                new_tour = new_tour1 + new_tour2 + new_tour3
                    
                if j == num_cities:
                    new_tour_score = NN_tour_distance - (dist_matrix[NN_tour[i - 1]][NN_tour[i]] + dist_matrix[NN_tour[j - 1]][NN_tour[0]])
                    new_tour_score += (dist_matrix[NN_tour[i - 1]][NN_tour[j - 1]] + dist_matrix[NN_tour[i]][NN_tour[0]])
                else:
                    new_tour_score = NN_tour_distance - (dist_matrix[NN_tour[i - 1]][NN_tour[i]] + dist_matrix[NN_tour[j - 1]][NN_tour[j]])
                    new_tour_score += (dist_matrix[NN_tour[i - 1]][NN_tour[j - 1]] + dist_matrix[NN_tour[i]][NN_tour[j]])
            
                if new_tour_score < NN_tour_distance:
                    NN_tour = new_tour
                    NN_tour_distance = new_tour_score
                    nflag = True
                j += 1
            i += 1 
    
    return NN_tour, NN_tour_distance

def three_opt(NN_tour, NN_tour_distance, node_nearest_neighbours):

    nflag = True
    while nflag == True:
        nflag = False
        i = 1
        while i < num_cities - 1 and nflag == False:
            j = i + 1
            while j < num_cities and nflag == False:
                k = j + 1
                while k <= num_cities and nflag == False:
                    if ((NN_tour[k - 1] not in node_nearest_neighbours[NN_tour[j - 1]] and NN_tour[k - 1] not in node_nearest_neighbours[NN_tour[i - 1]])
                    and (NN_tour[j - 1] not in node_nearest_neighbours[NN_tour[k - 1]] and NN_tour[j-1] not in node_nearest_neighbours[NN_tour[i - 1]]) 
                    and (NN_tour[i - 1] not in node_nearest_neighbours[NN_tour[k - 1]] and NN_tour[i-1] not in node_nearest_neighbours[NN_tour[j - 1]])):
                        k += 1
                        continue 
                
                    if k == num_cities:
                        dist0 = dist_matrix[NN_tour[i-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[0]]                            
                        dist1 = dist_matrix[NN_tour[i-1]][NN_tour[j-1]] + dist_matrix[NN_tour[j]][NN_tour[i]] + dist_matrix[NN_tour[k-1]][NN_tour[0]]                            
                        dist2 = dist_matrix[NN_tour[i-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[k-1]] + dist_matrix[NN_tour[j]][NN_tour[0]]                            
                        dist3 = dist_matrix[NN_tour[i-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[0]]                            
                        dist4 = dist_matrix[NN_tour[0]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[i-1]]
                                                   
                    else:
                        dist0 = dist_matrix[NN_tour[i-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[k]]                            
                        dist1 = dist_matrix[NN_tour[i-1]][NN_tour[j-1]] + dist_matrix[NN_tour[j]][NN_tour[i]] + dist_matrix[NN_tour[k-1]][NN_tour[k]]                            
                        dist2 = dist_matrix[NN_tour[i-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[k-1]] + dist_matrix[NN_tour[j]][NN_tour[k]]                            
                        dist3 = dist_matrix[NN_tour[i-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[k]]                            
                        dist4 = dist_matrix[NN_tour[k]][NN_tour[i]] + dist_matrix[NN_tour[j-1]][NN_tour[j]] + dist_matrix[NN_tour[k-1]][NN_tour[i-1]]

                    new_tour1 = NN_tour[0:i]
                    new_tour2 = NN_tour[i:j]
                    new_tour3 = NN_tour[j:k]
                    new_tour4 = NN_tour[k:]
                    new_tour5 = NN_tour[i:k]

                    if dist1 < dist0:
                        NN_tour_distance = NN_tour_distance - dist0 + dist1
                        NN_tour = new_tour1 + list(reversed(new_tour2)) + new_tour3 + new_tour4
                        nflag = True
                    elif dist2 < dist0:
                        NN_tour_distance = NN_tour_distance - dist0 + dist2
                        NN_tour = new_tour1 + new_tour2 + list(reversed(new_tour3)) + new_tour4
                        nflag = True
                    elif dist3 < dist0:
                        NN_tour_distance = NN_tour_distance - dist0 + dist3
                        tmp = new_tour3 + new_tour2
                        NN_tour = new_tour1 + tmp + new_tour4
                        nflag = True
                    elif dist4 < dist0: 
                        NN_tour_distance = NN_tour_distance - dist0 + dist4
                        NN_tour = new_tour1 + list(reversed(new_tour5)) + new_tour4
                        nflag = True
                    k += 1
                j += 1
            i += 1
    return NN_tour, NN_tour_distance



def ACO(max_it, num_ants, alpha, beta, p, w, num_NN, opt_num, local_ants):
    
    
    #represent our problem as a weighted graph G = (N(nodes), E(edges))
    #E holds a list of all edges organised according to the first city of the edge
    #E.g. E[5][8] holds info for edge between city/node 5 and 8: ([5, 8], pheromone level, distance)

    N = list(range(num_cities))
    E = []
    node_info = []

    for city in range(num_cities):
        node_info = []
        nodes_connected = N
            
        for node in nodes_connected:
            if node == city:
                node_info.append("NULL")
            else:
                edge_info = []
                edge_info.append([city, node])
                edge_info.append(0)
                edge_info.append(dist_matrix[city][node])
                node_info.append(edge_info)
        E.append(node_info)

    node_nearest_neighbours = []
    
    for node in E:
        edges_near = []
        sorted_edges = [x for x in node if x != "NULL"]
        sorted_edges = sorted(sorted_edges, key = lambda node: node[2])
        for i in range(num_NN):
            edges_near.append(sorted_edges[i][0][1])
        node_nearest_neighbours.append(edges_near)
    
    #nearest neighbour algorithm
    starting_city = random.randint(0, num_cities - 1)
    list_c = N
    list_c.remove(starting_city)
    NN_score = 0
    path = [starting_city, [starting_city], 0, list_c]
         
    flag = True
    while flag == True:
        best_city = -1
        best_city_distance = 10000000
        for city in path[3]:  
            if E[path[0]][city][2] < best_city_distance:
                best_city = city
                best_city_distance = E[path[0]][city][2]
        path[3].remove(best_city)
        path[2] += dist_matrix[path[0]][best_city]
        path[1].append(best_city)
        path[0] = best_city

        if path[3] == []:
            path[2] += dist_matrix[best_city][path[1][0]]
            NN_tour = path[2]
            flag = False
            
    #initial_deposit
    initial_deposit = num_ants/NN_tour
    for node in E:
            for edge in node:
                if edge == 'NULL':
                    continue
                edge[1] = initial_deposit

    #list of lists, each list in the list represents an ant: (current node, partial tour, tour distance, unvisited cities)
    ants = []

    for antid in range(num_ants):
        ant_to_add = []
        city_start = random.randint(0, num_cities - 1)
        ant_to_add.append(city_start)
        ant_to_add.append([city_start])
        ant_to_add.append(0)
        unvisited_cities = list(range(num_cities))
        unvisited_cities.remove(city_start)
        ant_to_add.append(unvisited_cities)
        ants.append(ant_to_add)

    iterations = 0
    best_tour = []
    best_tour_score = 100000000
    
    
    while iterations < max_it:   

        if(time.time() - start_time > 50.0):
            return best_tour, best_tour_score

        #reset ants after last iteration
        if iterations > 0:
            for ant in ants:
                start_cit = random.randint(0, num_cities - 1)
                ant[0] = start_cit
                ant[1] = [start_cit]
                ant[2] = 0
                unvisited_cit = list(range(num_cities))
                unvisited_cit.remove(start_cit)
                ant[3] = unvisited_cit

        i = 0
        while i < (num_cities-1):

            for ant in ants:
                pheromone = 0
                weightss = []
                fflag = True
                node_choices = []
                for node in node_nearest_neighbours[ant[0]]:
                    if node not in ant[3]:
                        continue
                    else:
                        pheromone = E[ant[0]][node][1]
                        if E[ant[0]][node][2] == 0:
                            inv_distance = 1
                        else:
                            inv_distance = 1 / E[ant[0]][node][2]
                        weightss.append((pheromone * alpha) * (inv_distance * beta))
                        node_choices.append(node)
                        fflag = False

                if fflag == True:
                    for edge in ant[3]:
                        pheromone = E[ant[0]][edge][1]
                        if E[ant[0]][edge][2] == 0:
                            inv_distance = 1
                        else:
                            inv_distance = 1 / E[ant[0]][edge][2]
                        weightss.append((pheromone * alpha) * (inv_distance * beta))
                    new_node = random.choices(ant[3], weights = weightss)
                else:
                    new_node = random.choices(node_choices, weights = weightss)

                ant[3].remove(new_node[0])
                ant[2] += dist_matrix[ant[0]][new_node[0]]
                ant[1] = ant[1] + new_node
                ant[0] = int(new_node[0])
                if ant[3] == []:
                    ant[2] += dist_matrix[ant[0]][ant[1][0]]
                    ant[0] = ant[1][0]
                    
            i += 1

        #pheromone decay
        upper_deposit = (1/p) * (1/best_tour_score)
        lowest_deposit = upper_deposit / (2*num_cities)

        for nodee in E:
            for edgee in nodee:
                if edgee == 'NULL':
                    continue
                elif edgee[1] < lowest_deposit:
                    continue
                edgee[1] = (1 - p) * edgee[1]

        #backtract and deposit pheromone
        for ant in ants[0:local_ants - 1]:
            if opt_num == 2:
                ant[1], ant[2] = two_opt(ant[1], ant[2], node_nearest_neighbours)
            else:
                ant[1], ant[2] = three_opt(ant[1], ant[2], node_nearest_neighbours)
            
            if ant[2] < best_tour_score:
                best_tour = ant[1]
                best_tour_score = ant[2]

        best_ant = [0, best_tour, best_tour_score, []]
        sorted_ants = sorted(ants, key = lambda ants: ants[2])

        sorted_ants.insert(0, best_ant)
        sorted_ants = sorted_ants[0:w]
    
        r = 0
        while r < w:
            j = 0
            ant = sorted_ants[r]
            while j < len(ant[1]) - 1:
                if j == 0 and (E[ant[1][-1]][ant[1][j]][1] < upper_deposit):
                    E[ant[1][-1]][ant[1][j]][1] += (w - r) * (1 / ant[2])

                elif E[ant[1][j]][ant[1][j + 1]][1] > upper_deposit:
                    j += 1
                    continue
                E[ant[1][j]][ant[1][j + 1]][1] += (w - r) * (1 / ant[2])

                j += 1
            r += 1

        iterations += 1

    return best_tour, best_tour_score


#ACO(max_it, num_ants, alpha, beta, p, w, length candidate lists, 2 or 3 opt, num ants local search applied to)
tour, tour_length = ACO(max_iterations, number_ants, alpha, beta, p, w, length_candidate_lists, two_or_three_opt, number_ants_locally_searched) 

############
############ YOUR CODE SHOULD NOW BE COMPLETE AND WHEN EXECUTION OF THIS PROGRAM 'skeleton.py'
############ REACHES THIS POINT, YOU SHOULD HAVE COMPUTED A TOUR IN THE RESERVED LIST VARIABLE 'tour', 
############ WHICH HOLDS A LIST OF THE INTEGERS FROM {0, 1, ..., 'num_cities' - 1}, AND YOU SHOULD ALSO
############ HOLD THE LENGTH OF THIS TOUR IN THE RESERVED INTEGER VARIABLE 'tour_length'.
############

############
############ YOUR TOUR WILL BE PACKAGED IN A TOUR FILE OF THE APPROPRIATE FORMAT AND THIS TOUR FILE,
############ WHOSE NAME WILL BE A MIX OF THE NAME OF THE CITY FILE, THE NAME OF THIS PROGRAM AND THE
############ CURRENT DATA AND TIME. SO, EVERY SUCCESSFUL EXECUTION GIVES A TOUR FILE WITH A UNIQUE
############ NAME AND YOU CAN RENAME THE ONES YOU WANT TO KEEP LATER.
############

############
############ DO NOT TOUCH OR ALTER THE CODE BELOW THIS POINT! YOU HAVE BEEN WARNED!
############

flag = "good"
length = len(tour)
for i in range(0, length):
    if isinstance(tour[i], int) == False:
        flag = "bad"
    else:
        tour[i] = int(tour[i])
if flag == "bad":
    print("*** error: Your tour contains non-integer values.")
    sys.exit()
if isinstance(tour_length, int) == False:
    print("*** error: The tour-length is a non-integer value.")
    sys.exit()
tour_length = int(tour_length)
if len(tour) != num_cities:
    print("*** error: The tour does not consist of " + str(num_cities) + " cities as there are, in fact, " + str(len(tour)) + ".")
    sys.exit()
flag = "good"
for i in range(0, num_cities):
    if not i in tour:
        flag = "bad"
if flag == "bad":
    print("*** error: Your tour has illegal or repeated city names.")
    sys.exit()
check_tour_length = 0
for i in range(0, num_cities - 1):
    check_tour_length = check_tour_length + dist_matrix[tour[i]][tour[i + 1]]
check_tour_length = check_tour_length + dist_matrix[tour[num_cities - 1]][tour[0]]
if tour_length != check_tour_length:
    flag = print("*** error: The length of your tour is not " + str(tour_length) + "; it is actually " + str(check_tour_length) + ".")
    sys.exit()
print("You, user " + my_user_name + ", have successfully built a tour of length " + str(tour_length) + "!")

local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]
output_file_name = script_name[0:len(script_name) - 3] + "_" + input_file[0:len(input_file) - 4] + "_" + output_file_time + ".txt"

f = open(output_file_name,'w')
f.write("USER = " + my_user_name + " (" + my_first_name + " " + my_last_name + "),\n")
f.write("ALGORITHM CODE = " + algorithm_code + ", NAME OF CITY-FILE = " + input_file + ",\n")
f.write("SIZE = " + str(num_cities) + ", TOUR LENGTH = " + str(tour_length) + ",\n")
f.write(str(tour[0]))
for i in range(1,num_cities):
    f.write("," + str(tour[i]))
f.write(",\nNOTE = " + added_note)
f.close()
print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")