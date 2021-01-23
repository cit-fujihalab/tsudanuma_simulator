import networkx as nx
import numpy as np
import math
import copy
import pprint

class Car:
  def __init__(self, orig_node_id, dest_node_id, dest_lane_id, shortest_path, current_lane_id, DG):
    self.orig_node_id  = orig_node_id
    self.dest_node_id  = dest_node_id
    self.dest_lane_id = dest_lane_id
    self.shortest_path = shortest_path
    self.current_lane_id =  current_lane_id
    self.current_sp_index = 0
    self.current_speed = 0.0
    self.current_start_node = []
    self.current_position = []
    self.current_end_node = []
    self.obstacles_info_list = []
    self.current_distance = 0.0
    self.elapsed_time = 0
    self.moving_distance = 0
    self.goal_arrived = False
    self.DG_copied = copy.deepcopy(DG)
    self.opportunistic_communication_frag = True
    self.aco_frag = False
    self.pheromone = 1

  def init(self, DG):
    current_start_node_id = self.shortest_path[ self.current_sp_index ]
    self.current_start_node = DG.nodes[ current_start_node_id ]["pos"]
    self.current_position = DG.nodes[ current_start_node_id ]["pos"]
    current_end_node_id = self.shortest_path[ self.current_sp_index+1]
    self.current_end_node = DG.nodes[ current_end_node_id ]["pos"]
    current_edge_attributes = DG.get_edge_data(current_start_node_id, current_end_node_id)
    self.current_max_speed = current_edge_attributes["speed"]
    self.current_distance = current_edge_attributes["weight"]

  # Optimal Velocity Function to determine the current speed
  def V(self, inter_car_distance):
    return 0.5*self.current_max_speed*(np.tanh(inter_car_distance-2) + np.tanh(2))

  def update_current_speed(self, sensitivity, inter_car_distance):
    self.current_speed += sensitivity*( self.V(inter_car_distance) - self.current_speed )

  def move(self, x_y_dic, edges_cars_dic, sensitivity, lane_node_dic, edge_length_dic, edge_lanes_list, obstacle_node_id_list, time):
    self.elapsed_time += 1

    direction_x = self.current_end_node[0] - self.current_position[0]
    direction_y = self.current_end_node[1] - self.current_position[1]
    arg = math.atan2(direction_y, direction_x)

    if np.sqrt((self.current_position[0] - self.current_end_node[0])**2 + (self.current_position[1] - self.current_end_node[1])**2) < self.current_speed: # to arrive at the terminal of edge

      if self.shortest_path[self.current_sp_index] in lane_node_dic:
        self.moving_distance += edge_length_dic[lane_node_dic[self.shortest_path[self.current_sp_index]]]

      self.current_sp_index += 1

      if self.current_sp_index >= len(self.shortest_path)-1: # arrived at the goal
        self.goal_arrived = True
        x_new = self.current_end_node[0]
        y_new = self.current_end_node[1]

        current_start_node_id = self.shortest_path[ self.current_sp_index-1 ]
        current_end_node_id = self.shortest_path[ self.current_sp_index ]
        self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]

        car_forward_index = edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self)
        car_forward_pt = edges_cars_dic[(current_start_node_id, current_end_node_id)][car_forward_index]
        diff_dist = 50.0

        edges_cars_dic[ (current_start_node_id, current_end_node_id) ].remove( self )

      else: # lane change
        if self.shortest_path[self.current_sp_index] in lane_node_dic:
          self.moving_distance += edge_length_dic[lane_node_dic[self.shortest_path[self.current_sp_index]]]
        x_new = self.current_end_node[0]
        y_new = self.current_end_node[1]

        current_start_node_id = self.shortest_path[ self.current_sp_index-1 ]
        current_end_node_id = self.shortest_path[ self.current_sp_index ]
        edges_cars_dic[ (current_start_node_id, current_end_node_id) ].remove( self )

        counter = 0
        while True:
          try:
            counter += 1
            if counter == 100:
              break
            self.shortest_path = nx.dijkstra_path(self.DG_copied, self.shortest_path[ self.current_sp_index ], self.dest_node_id)
            break

          except Exception:
            self.dest_lane_id = np.random.randint(len(edge_lanes_list))
            self.dest_node_id = x_y_dic[(edge_lanes_list[self.dest_lane_id].node_x_list[-1], edge_lanes_list[self.dest_lane_id].node_y_list[-1])]

            while self.dest_node_id in obstacle_node_id_list or self.current_lane_id == self.dest_lane_id:
              self.dest_lane_id = np.random.randint(len(edge_lanes_list))
              self.dest_node_id = x_y_dic[(edge_lanes_list[self.dest_lane_id].node_x_list[-1], edge_lanes_list[self.dest_lane_id].node_y_list[-1])]

        self.current_sp_index = 0
        current_start_node_id = self.shortest_path[self.current_sp_index]
        self.current_start_node = self.DG_copied.nodes[current_start_node_id]["pos"]
        self.current_position = self.DG_copied.nodes[current_start_node_id]["pos"]
        current_end_node_id = self.shortest_path[self.current_sp_index + 1]
        self.current_end_node = self.DG_copied.nodes[current_end_node_id]["pos"]
        edges_cars_dic[(current_start_node_id, current_end_node_id)].append(self)

        #if self.aco_frag == True:
         # aco()

        if edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self) > 0:
          car_forward_index = edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self) - 1
          car_forward_pt = edges_cars_dic[(current_start_node_id, current_end_node_id)][car_forward_index]
          diff_dist = 50

        else:
          car_forward_index = edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self)
          car_forward_pt = edges_cars_dic[(current_start_node_id, current_end_node_id)][car_forward_index]
          diff_dist = 50.0

    else: # move to the terminal of edge
      x_new = self.current_position[0] + self.current_speed*np.cos(arg)
      y_new = self.current_position[1] + self.current_speed*np.sin(arg)
      self.current_position = [x_new, y_new]
      current_start_node_id = self.shortest_path[ self.current_sp_index ]
      current_end_node_id = self.shortest_path[ self.current_sp_index+1 ]

      if edges_cars_dic[ (current_start_node_id, current_end_node_id) ].index( self ) > 0:
        car_forward_index = edges_cars_dic[ (current_start_node_id, current_end_node_id) ].index( self ) - 1
        car_forward_pt = edges_cars_dic[ (current_start_node_id, current_end_node_id) ][ car_forward_index ]
        diff_dist = np.sqrt( (car_forward_pt.current_position[0] - self.current_position[0])**2 + (car_forward_pt.current_position[1] - self.current_position[1])**2 )

      else:
        car_forward_index = edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self)
        car_forward_pt = edges_cars_dic[(current_start_node_id, current_end_node_id)][car_forward_index]
        diff_dist = 50.0
      self.update_current_speed(sensitivity, diff_dist)

      try:
        self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]
      except KeyError:
        pass

    if time > 1000:
     try:
      self.current_sp_index+=1
      self.goal_arrived = True
      x_new = self.current_end_node[0]
      y_new = self.current_end_node[1]

      current_start_node_id = self.shortest_path[self.current_sp_index - 1]
      current_end_node_id = self.shortest_path[self.current_sp_index]
      self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]

      car_forward_index = edges_cars_dic[(current_start_node_id, current_end_node_id)].index(self)
      car_forward_pt = edges_cars_dic[(current_start_node_id, current_end_node_id)][car_forward_index]
      diff_dist = 50
      edges_cars_dic[(current_start_node_id, current_end_node_id)].remove(self)
     except Exception:
      pass
    return x_new, y_new, self.goal_arrived, car_forward_pt, diff_dist

  def U_turn(self, edges_cars_dic,lane_node_dic, edge_lanes_list, x_y_dic, obstacle_node_id_list, road_segment_dic):
    self.current_sp_index += 1

    x_new = self.current_end_node[0]
    y_new = self.current_end_node[1]

    current_start_node_id = self.shortest_path[self.current_sp_index - 1]
    current_end_node_id = self.shortest_path[self.current_sp_index]
    edges_cars_dic[(current_start_node_id, current_end_node_id)].remove(self)
    pre_start_node_id = current_start_node_id
    pre_end_node_id = current_end_node_id

    if current_end_node_id not in self.obstacles_info_list:
      self.obstacles_info_list.append(current_end_node_id)

    self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]
    current_start_node_id = x_y_dic[(edge_lanes_list[road_segment_dic[self.current_lane_id]].node_x_list[0], edge_lanes_list[road_segment_dic[self.current_lane_id]].node_y_list[0])]
    current_end_node_id = x_y_dic[(edge_lanes_list[road_segment_dic[self.current_lane_id]].node_x_list[-1], edge_lanes_list[road_segment_dic[self.current_lane_id]].node_y_list[-1])]
    self.current_start_node = self.DG_copied.nodes[current_start_node_id]["pos"]
    self.current_position = self.DG_copied.nodes[current_start_node_id]["pos"]
    self.current_end_node = self.DG_copied.nodes[current_end_node_id]["pos"]

    if self.DG_copied.has_edge(pre_start_node_id, pre_end_node_id) == True:
      self.DG_copied.remove_edge(pre_start_node_id, pre_end_node_id)

    while True:
      try:
        self.shortest_path = nx.dijkstra_path(self.DG_copied, current_start_node_id, self.dest_node_id)
        break

      except Exception:
        self.dest_lane_id = np.random.randint(len(edge_lanes_list))
        self.dest_node_id = x_y_dic[(edge_lanes_list[self.dest_lane_id].node_x_list[-1], edge_lanes_list[self.dest_lane_id].node_y_list[-1])]

        while self.dest_node_id in obstacle_node_id_list or self.current_lane_id == self.dest_lane_id:
          self.dest_lane_id = np.random.randint(len(edge_lanes_list))
          self.dest_node_id = x_y_dic[(edge_lanes_list[self.dest_lane_id].node_x_list[-1], edge_lanes_list[self.dest_lane_id].node_y_list[-1])]

    self.current_sp_index = 0

    current_start_node_id = self.shortest_path[self.current_sp_index]
    self.current_start_node = self.DG_copied.nodes[current_start_node_id]["pos"]
    self.current_position = self.DG_copied.nodes[current_start_node_id]["pos"]
    current_end_node_id = self.shortest_path[self.current_sp_index + 1]
    self.current_end_node = self.DG_copied.nodes[current_end_node_id]["pos"]
    current_edge_attributes = self.DG_copied.get_edge_data(current_start_node_id, current_end_node_id)
    self.current_max_speed = current_edge_attributes["speed"]
    self.current_distance = current_edge_attributes["weight"]
    edges_cars_dic[(current_start_node_id, current_end_node_id)].append(self)

    return x_new,y_new

  def opportunistic(self, edge_lanes_list,edges_cars_dic, lane_node_dic, x_y_dic, road_segment_dic):
    try:
      self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]

      for other_car in edges_cars_dic[(x_y_dic[(edge_lanes_list[self.current_lane_id].node_x_list[0], edge_lanes_list[self.current_lane_id].node_y_list[0])],
                                       x_y_dic[(edge_lanes_list[self.current_lane_id].node_x_list[1], edge_lanes_list[self.current_lane_id].node_y_list[1])])]:
        if other_car.__class__.__name__ == "Car" and other_car.opportunistic_communication_frag == True:
          for i in other_car.obstacles_info_list:
            if i not in self.obstacles_info_list:
              self.obstacles_info_list.append(i)
              a = x_y_dic[(edge_lanes_list[lane_node_dic[self.obstacles_info_list[-1]]].node_x_list[0],edge_lanes_list[lane_node_dic[self.obstacles_info_list[-1]]].node_y_list[0])]
              if self.DG_copied.has_edge(a, self.obstacles_info_list[-1]) == True:
                self.DG_copied.remove_edge(a, self.obstacles_info_list[-1])

      try:
        for oc_car in edges_cars_dic[(x_y_dic[(edge_lanes_list[road_segment_dic[self.current_lane_id]].node_x_list[0], edge_lanes_list[road_segment_dic[self.current_lane_id]].node_y_list[0])],
                                      x_y_dic[(edge_lanes_list[road_segment_dic[self.current_lane_id]].node_x_list[1], edge_lanes_list[road_segment_dic[self.current_lane_id]].node_y_list[1])])]:
          if oc_car.__class__.__name__ == "Car" and oc_car.opportunistic_communication_frag == True:
            for i in oc_car.obstacles_info_list:
              if i not in self.obstacles_info_list:
                self.obstacles_info_list.append(i)
                a = x_y_dic[(edge_lanes_list[lane_node_dic[self.obstacles_info_list[-1]]].node_x_list[0],edge_lanes_list[lane_node_dic[self.obstacles_info_list[-1]]].node_y_list[0])]
                if self.DG_copied.has_edge(a, self.obstacles_info_list[-1]) == True:
                  self.DG_copied.remove_edge(a, self.obstacles_info_list[-1])

      except Exception:
        pass
    except KeyError:
      pass

  def ant(self, lane_dic2,lane_node_dic, x_y_dic, phero):
    counter = 0
    DG_copied2 = copy.deepcopy(self.DG_copied)
    DG_copied2.remove_edge(self.shortest_path[self.current_sp_index + 1],self.shortest_path[self.current_sp_index+2])
    sp = self.shortest_path
    sp_index = self.current_sp_index
    while True:
      try:
        counter += 1
        self.shortest_path = nx.dijkstra_path(DG_copied2, x_y_dic[(self.current_position[0], self.current_position[1])],self.dest_node_id)
        self.current_sp_index = 0
        if lane_dic2[lane_node_dic[self.shortest_path[self.current_sp_index + 1]]].pheromone < phero*10: # 次のノードのフェロモン量が一定以下(渋滞でない)の場合break
          print(self.shortest_path)
          break

        elif lane_dic2[lane_node_dic[self.shortest_path[self.current_sp_index + 1]]].pheromone >= phero:
          DG_copied2.remove_edge(self.shortest_path[self.current_sp_index + 1],self.shortest_path[self.current_sp_index + 2])

      except Exception:
        if counter == 10:
          self.shortest_path = sp
          self.current_sp_index = sp_index
          break

    if self.shortest_path != sp:
      current_start_node_id = self.shortest_path[self.current_sp_index]
      self.current_start_node = self.DG_copied.nodes[current_start_node_id]["pos"]
      self.current_position = self.DG_copied.nodes[current_start_node_id]["pos"]
      current_end_node_id = self.shortest_path[self.current_sp_index + 1]
      self.current_end_node = self.DG_copied.nodes[current_end_node_id]["pos"]
      current_edge_attributes = self.DG_copied.get_edge_data(current_start_node_id, current_end_node_id)
      self.current_max_speed = current_edge_attributes["speed"]
      # = current_edge_attributes["weight"]
      #edges_cars_dic[(current_start_node_id, current_end_node_id)].append(self)
      self.current_lane_id = lane_node_dic[self.shortest_path[self.current_sp_index]]
      print("path change")
