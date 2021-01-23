#!/usr/bin/env python3
# coding: utf-8
#import matplotlib
#matplotlib.use('Agg')

import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from math import sin, cos, acos, radians
from PIL import Image, ImageOps
import smopy
import json
import gc
import copy
import csv
import datetime

from car import Car
from lane import Lane
from obstacle import Obstacle
from road_segment import RoadSegment

earth_rad = 6378.137
np.random.seed(12345)

infilename = "tsudanuma.net.xml"
png_infilename = "tsudanuma.png"
filename_geojson = "tsudanuma.geojson"

number_of_cars = 5000
number_of_obstacles = 400
oppcomm_rate = 0.0
num_of_division = 4

sensitivity = 1.0

def read_parse_netxml(infilename):
  tree = ET.parse(infilename)
  root = tree.getroot()

  return root

def get_map_smopy():
  infile = open(filename_geojson, "r")
  data_dic = json.load(infile)
  max_lon = -180.0; min_lon = 180.0
  max_lat = -90.0; min_lat = 90.0
  for l in data_dic["features"][0]["geometry"]["coordinates"][0]:
    #print(l)
    if max_lon < float(l[0]):
      max_lon = float(l[0])
    if max_lat < float(l[1]):
      max_lat = float(l[1])
    if min_lon > float(l[0]):
      min_lon = float(l[0])
    if min_lat > float(l[1]):
      min_lat = float(l[1])

  lon_lat_tuple = (min_lat, min_lon, max_lat, max_lon)

  z=17
  smopy_map = smopy.Map(lon_lat_tuple, tileserver="https://tile.openstreetmap.org/{z}/{x}/{y}.png", tilesize=256, maxtiles=16, z=z)
  print("got map")

  px_min_lon, px_min_lat = smopy_map.to_pixels( lat=lon_lat_tuple[0], lon=lon_lat_tuple[1] )
  px_max_lon, px_max_lat = smopy_map.to_pixels( lat=lon_lat_tuple[2], lon=lon_lat_tuple[3] )

  x0 = min(px_max_lon, px_min_lon)
  x1 = max(px_max_lon, px_min_lon)
  y0 = min(px_max_lat, px_min_lat)
  y1 = max(px_max_lat, px_min_lat)

  smopy_map.save_png(png_infilename)

  return smopy_map, x0, x1, y0, y1, lon_lat_tuple

def create_road_network(root, smopy_map):
  def get_boundary(root):
    for child in root:
      if child.tag == "location":
        convBoundary = list(map(float,child.attrib["convBoundary"].split(",")))
        origBoundary = list(map(float,child.attrib["origBoundary"].split(",")))
      return convBoundary, origBoundary

  def calculate_coordinates(convBoundary, origBoundary, node_x, node_y):
    orig_per_conv_X = abs(origBoundary[0] - origBoundary[2]) / abs(convBoundary[0] - convBoundary[2])
    orig_per_conv_Y = abs(origBoundary[1] - origBoundary[3]) / abs(convBoundary[1] - convBoundary[3])

    node_x = origBoundary[0] + (node_x * orig_per_conv_X)
    node_y = origBoundary[1] + (node_y * orig_per_conv_Y)
    return node_x, node_y

  def is_not_roadway(child):
    childs = str(child.attrib).split(",")
    for ch in childs:
      if "railway" in ch:
        return True
      if "highway.cycleway" in ch:
        return True
      if "highway.footway" in ch:
        return True
      if "highway.living_street" in ch:
        return True
      if "highway.path" in ch:
        return True
      if "highway.pedestrian" in ch:
        return True
      if "highway.step" in ch:
        return True
    return False

  lane_dic = {}
  lane_dic2 = {}
  x_y_dic = {}
  edge_length_dic = {}
  lane_node_dic = {}
  l_n_dic = {}
  node_id = 0
  lane_id = 0
  DG = nx.DiGraph()
  edge_lanes_list = []

  convBoundary, origBoundary = get_boundary(root)

  top = origBoundary[3]
  bottom = origBoundary[1]
  leftmost = origBoundary[0]
  rightmost = origBoundary[2]
  x_of_divided_area = abs(leftmost - rightmost) / num_of_division
  y_of_divided_area = abs(top - bottom) / num_of_division

  node_id_to_index = {}
  index_to_node_id = {}
  node_id_to_coordinate = {}

  for child in root:
    if child.tag == "edge":
      if is_not_roadway(child):
        continue

      lane = Lane()

      if "from" in child.attrib and "to" in child.attrib:
        lane.add_from_to(child.attrib["from"], child.attrib["to"])

      for child2 in child:

        try:
          data_list = child2.attrib["shape"].split(" ")
        except:  # except param
          continue

        node_id_list = []
        node_x_list = []; node_y_list = []
        distance_list = []
        data_counter = 0

        for data in data_list:
          node_lon, node_lat = calculate_coordinates(convBoundary, origBoundary, float(data.split(",")[0]), float(data.split(",")[1]))

          node_x, node_y = smopy_map.to_pixels(node_lat,node_lon)

          index_x = int(abs(leftmost - node_lon) // x_of_divided_area)
          index_y = int(abs(top - node_lat) // y_of_divided_area)

          #緯度経度(xml,geojson)の誤差?によりindex==num_of_divisionとなる場合があるため、エリア内に収まるように調整する
          if not 0 <= index_x < num_of_division:
            if index_x >= num_of_division:
              index_x = num_of_division - 1
            else:
              index_x = 0
          if not 0 <= index_y < num_of_division:
            if index_y >= num_of_division:
              index_y = num_of_division - 1
            else:
              index_y = 0

          node_id_to_index[node_id] = (index_x, index_y)
          if (index_x, index_y) not in index_to_node_id.keys():
            index_to_node_id[(index_x, index_y)] = [node_id]
          else:
            index_to_node_id[(index_x, index_y)].append(node_id)

          node_id_to_coordinate[node_id] = {
            "longitude": node_lon,
            "latitude": node_lat
          }

          node_x_list.append( node_x )
          node_y_list.append( node_y )

          if (node_x, node_y) not in x_y_dic.keys():
            node_id_list.append(node_id)
            DG.add_node(node_id, pos=(node_x, node_y))
            x_y_dic[ (node_x, node_y) ] = node_id
            node_id += 1

          else:
            node_id_list.append( x_y_dic[ (node_x, node_y) ] )

          if data_counter >= 1:
            distance_list.append(np.sqrt((node_x - old_node_x) ** 2 + (node_y - old_node_y) ** 2))
          old_node_x = node_x
          old_node_y = node_y
          data_counter += 1
        for i in range(len(node_id_list) - 1):
            DG.add_edge(node_id_list[i], node_id_list[i + 1], weight=distance_list[i], color="black",speed=float(child2.attrib["speed"]))  # calculate weight here

        if "from" in child.attrib and "to" in child.attrib:
            lane.set_others(float(child2.attrib["speed"]), node_id_list, node_x_list, node_y_list)
            edge_lanes_list.append(lane)  # to modify here
            lane_dic[lane] = lane_id
            lane_dic2[lane_id] = lane
            edge_length_dic[lane_id] = float(child2.attrib["length"])
            for i in range(len(node_x_list)):
              l_n_dic[(x_y_dic[node_x_list[i], node_y_list[i]])] = lane_id
            lane_id += 1


  scc = nx.strongly_connected_components(DG)
  largest_scc = True
  for c in sorted(scc, key=len, reverse=True):
      if largest_scc == True:
          largest_scc = False
      else:
          c_list = list(c)
          for i in range(len(c_list)):
              DG.remove_node(c_list[i])
              for lane in edge_lanes_list:
                  if lane.node_id_list[0] == int(c_list[i]) or lane.node_id_list[1] == int(c_list[i]):
                    edge_lanes_list.remove(lane)

  lane_id = 0
  for lane in edge_lanes_list:
    for k,v in l_n_dic.items():
      if v == lane_dic[lane]:
        lane_node_dic[k] = lane_id
    lane_id += 1

  return x_y_dic, DG, edge_lanes_list, node_id_to_index, index_to_node_id, node_id_to_coordinate, edge_length_dic, lane_node_dic, lane_dic2

def create_road_segments(edge_lanes_list):
  road_segments_dic = {}
  road_segments_list = []
  for i in range(len(edge_lanes_list)-1):
    for j in range(i+1, len(edge_lanes_list)):
      if edge_lanes_list[i].from_id == edge_lanes_list[j].to_id and edge_lanes_list[i].to_id == edge_lanes_list[j].from_id:
        road_segments_list.append(RoadSegment(edge_lanes_list[i], edge_lanes_list[j]))
        break
  for i in range(len(edge_lanes_list)):
    for j in range(len(edge_lanes_list)):
      if edge_lanes_list[i].from_id == edge_lanes_list[j].to_id and edge_lanes_list[i].to_id == edge_lanes_list[j].from_id:
        road_segments_dic[i] = j
  return road_segments_list, road_segments_dic

def find_OD_node_and_lane():

  origin_lane_id = np.random.randint(len(edge_lanes_list))
  destination_lane_id = origin_lane_id
  while origin_lane_id == destination_lane_id:
    destination_lane_id = np.random.randint(len(edge_lanes_list))

  origin_node_id = x_y_dic[(edge_lanes_list[origin_lane_id].node_x_list[0], edge_lanes_list[origin_lane_id].node_y_list[0])]
  destination_node_id = x_y_dic[(edge_lanes_list[destination_lane_id].node_x_list[-1], edge_lanes_list[destination_lane_id].node_y_list[-1])]

  while origin_node_id in obstacle_node_id_list:
    origin_lane_id = np.random.randint(len(edge_lanes_list))
    origin_node_id = x_y_dic[(edge_lanes_list[origin_lane_id].node_x_list[0], edge_lanes_list[origin_lane_id].node_y_list[0])]

  while destination_node_id in obstacle_node_id_list or origin_lane_id == destination_lane_id:
      destination_lane_id = np.random.randint(len(edge_lanes_list))
      destination_node_id = x_y_dic[(edge_lanes_list[destination_lane_id].node_x_list[-1], edge_lanes_list[destination_lane_id].node_y_list[-1])]

  return origin_lane_id, destination_lane_id, origin_node_id, destination_node_id


def find_obstacle_lane_and_node():
  while True:
    obstacle_lane_id = np.random.randint(len(edge_lanes_list))
    obstacle_node_id = x_y_dic[(edge_lanes_list[obstacle_lane_id].node_x_list[-1], edge_lanes_list[obstacle_lane_id].node_y_list[-1])]
    try:
      if x_y_dic[(edge_lanes_list[road_segment_dic[obstacle_lane_id]].node_x_list[-1], edge_lanes_list[road_segment_dic[obstacle_lane_id]].node_y_list[-1])] not in obstacle_node_id_list and obstacle_node_id not in obstacle_node_id_list:
        break
    except Exception:
      pass

  obstacle_node_id_list.append(obstacle_node_id)
  pair_node_id_list.append(x_y_dic[(edge_lanes_list[obstacle_lane_id].node_x_list[0], edge_lanes_list[obstacle_lane_id].node_y_list[0])])

  return obstacle_lane_id, obstacle_node_id

def init():
  line1.set_data([], [])
  line2.set_data([], [])
  title.set_text("Simulation step: 0")
  return line1, line2, title,

def animate(time):
  global xdata,ydata,obstacle_x,obstacle_y
  global goal_time_list, moving_distance_list
  xdata = [];ydata = []
  phero = (number_of_cars / 100)  # 全体の車両の1%
  print("########## step " + str(time) + " ##########", datetime.datetime.now(), len(cars_list) - number_of_obstacles)

  for car in cars_list:
    if car.__class__.__name__ == 'Car':

      if car.aco_frag == True:
        for lane in lane_dic2:
          if lane == car.current_lane_id:
            try:
              if x_y_dic[(car.current_position[0], car.current_position[1])] == x_y_dic[(edge_lanes_list[lane_node_dic[car.shortest_path[car.current_sp_index + 1]]].node_x_list[0],edge_lanes_list[lane_node_dic[car.shortest_path[car.current_sp_index + 1]]].node_y_list[0])]:

                if car.pheromone != 0:
                  lane_dic2[lane].pheromone += 1  # フェロモンの散布
                  car.pheromone -= 1

                if lane_dic2[lane].pheromone >= phero:
                  lane_dic2[lane].pheromone = phero  # フェロモンの最大値を指定

            except Exception:
              pass

            try:
              if x_y_dic[(car.current_position[0], car.current_position[1])] == x_y_dic[(edge_lanes_list[lane_node_dic[car.shortest_path[car.current_sp_index + 1]]].node_x_list[1],edge_lanes_list[lane_node_dic[car.shortest_path[car.current_sp_index + 1]]].node_y_list[1])]:
                if car.shortest_path[car.current_sp_index + 2] != car.shortest_path[-1] and lane_dic2[lane_node_dic[car.shortest_path[car.current_sp_index + 1]]].pheromone == phero:
                  #print("aco start")
                  car.ant(lane_dic2, lane_node_dic, x_y_dic, phero)

                car.pheromone += 1
                lane_dic2[lane].pheromone -= 1  # フェロモン蒸発
            except Exception:
              pass

      if car.opportunistic_communication_frag == True:
        #print("opportunistic communication start")
        car.opportunistic(edge_lanes_list,edges_cars_dic, lane_node_dic, x_y_dic, road_segment_dic)

      x_new, y_new, goal_arrived_flag, car_forward_pt, diff_dist = car.move(x_y_dic, edges_cars_dic, sensitivity, lane_node_dic, edge_length_dic, edge_lanes_list, obstacle_node_id_list, time)

      if car.goal_arrived == True:
        if time < 1000:
          goal_time_list.append(car.elapsed_time)
          moving_distance_list.append(round(car.moving_distance, 1))
        cars_list.remove(car)

      # TODO: if the car encounters road closure, it U-turns.
      if car_forward_pt.__class__.__name__ != "Car" and diff_dist <= 20:
        x_new, y_new = car.U_turn(edges_cars_dic, lane_node_dic, edge_lanes_list, x_y_dic, obstacle_node_id_list, road_segment_dic)

      xdata.append(x_new)
      ydata.append(y_new)

  obstacle_x = [];obstacle_y = []
  for obstacle in obstacles_list:
    x_new, y_new = obstacle.move()
    obstacle_x.append(x_new)
    obstacle_y.append(y_new)

  if len(cars_list) - number_of_obstacles == 0:

    print("Total simulation step: " + str(time - 1))
    print("### End of simulation ###")
    plt.clf()

    plt.hist(moving_distance_list, bins=50, rwidth=0.9, color='b')
    plt.savefig("総移動距離 " + infilename + " oppcommrate=" + str(oppcomm_rate) + "cars" + str(number_of_cars) + "obstacles" + str(number_of_obstacles) + ".png")
    plt.clf()

    plt.hist(goal_time_list, bins=50, rwidth=0.9, color='b')
    plt.savefig("ゴールタイム " + infilename + " oppcommrate=" + str(oppcomm_rate) + "cars" + str(number_of_cars) + "obstacles" + str(number_of_obstacles) + ".png")
    plt.clf()

    with open("result " + infilename + " oppcommrate=" + str(oppcomm_rate) + "cars" + str(number_of_cars) + "obstacles" + str(number_of_obstacles) + ".csv", 'w', newline='') as f:
      writer = csv.writer(f)

      for i in range(len(goal_time_list)):
        writer.writerow([goal_time_list[i], moving_distance_list[i]])
    sys.exit(0)  # end of simulation, exit.

  line1.set_data(xdata, ydata)
  line2.set_data(obstacle_x, obstacle_y)
  title.set_text("Simulation step: " + str(time) + ";  # of cars: " + str(len(cars_list) - number_of_obstacles))

  return line1, line2, title,

# Optimal Velocity Function
def V(b, current_max_speed):
  return 0.5*current_max_speed*(np.tanh(b-2) + np.tanh(2))

##### main #####
if __name__ == "__main__":

  smopy_map, x0, x1, y0, y1, lon_lat_tuple = get_map_smopy()

  root = read_parse_netxml(infilename)
  print("### create road network started ###")
  x_y_dic, DG, edge_lanes_list, node_id_to_index, index_to_node_id, node_id_to_coordinate, edge_length_dic, lane_node_dic, lane_dic2 = create_road_network(root,smopy_map)
  print("### create road network ended ###")

  #pprint.pprint(lane_node_dic)
  print("### create road segments started ###")
  road_segments_list, road_segment_dic = create_road_segments(edge_lanes_list)
  print("### create road segments ended ###")

  edges_all_list = DG.edges()
  edges_cars_dic = {}

  for item in edges_all_list:
    edges_cars_dic[item] = []

  obstacles_list = []
  obstacle_node_id_list = []
  pair_node_id_list = []
  cars_list = []
  goal_time_list = []
  moving_distance_list = []

  # create obstacles
  print("### create obstacles started ###")
  while True:
    for i in range(number_of_obstacles):
      obstacle_lane_id, obstacle_node_id = find_obstacle_lane_and_node()
      obstacle = Obstacle(obstacle_node_id, obstacle_lane_id)
      obstacle.init(DG)
      obstacles_list.append(obstacle)
      cars_list.append(obstacle)
      edges_cars_dic[(edge_lanes_list[obstacle_lane_id].node_id_list[0], edge_lanes_list[obstacle_lane_id].node_id_list[1])].append(obstacle)
    if nx.is_weakly_connected(DG) == True:
      break
  print("### create obstacles ended ###")

  print("### create cars started ###")
  # create cars
  DG_copied2 = copy.deepcopy(DG)
  for i in range(len(obstacle_node_id_list)):
    if DG_copied2.has_edge(pair_node_id_list[i],obstacle_node_id_list[i]):
      DG_copied2.remove_edge(pair_node_id_list[i],obstacle_node_id_list[i])

  for i in range(number_of_cars):
    # Reference: https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html
    origin_lane_id, destination_lane_id, origin_node_id, destination_node_id = find_OD_node_and_lane()
    while True:
      try:
        shortest_path = nx.dijkstra_path(DG_copied2, origin_node_id, destination_node_id)
        if len(shortest_path) > 1:
          break

        elif len(shortest_path) == 1:
          origin_lane_id, destination_lane_id, origin_node_id, destination_node_id = find_OD_node_and_lane()

      except Exception:
        origin_lane_id, destination_lane_id, origin_node_id, destination_node_id = find_OD_node_and_lane()

    shortest_path = nx.dijkstra_path(DG, origin_node_id, destination_node_id)
    car = Car(origin_node_id, destination_node_id, destination_lane_id, shortest_path, origin_lane_id, DG)
    car.init(DG)  # initialization of car settings
    cars_list.append(car)
    edges_cars_dic[(edge_lanes_list[origin_lane_id].node_id_list[0], edge_lanes_list[origin_lane_id].node_id_list[1])].append(car)
    if oppcomm_rate * number_of_cars < i:
      car.opportunistic_communication_frag = False
  print("### create cars ended ###")

  fig = plt.figure()
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(x0, x1), ylim=(y0, y1))

  xdata = [];ydata = []
  for i in range(len(cars_list)):
    xdata.append( cars_list[i].current_position[0] )
    ydata.append( cars_list[i].current_position[1] )
  obstacle_x = []; obstacle_y = []
  for i in range(len(obstacles_list)):
    obstacle_x.append(obstacles_list[i].current_position[0])
    obstacle_y.append(obstacles_list[i].current_position[1])

  line1, = plt.plot([], [], color="green", marker="s", linestyle="", markersize=3)
  line2, = plt.plot([], [], color="red", marker="s", linestyle="", markersize=3)
  title = ax.text(20.0, -20.0, "", va="center")

  print("### map image loading ###")
  img = Image.open(png_infilename)
  img_list = np.asarray(img)
  plt.imshow(img_list)
  print("### map image loaded ###")

  ax.invert_yaxis()
  gc.collect()

  print("### Start of simulation ###")
  ani = FuncAnimation(fig, animate, frames=range(50000), init_func=init, blit=True, interval=10)
  #ani.save("tsudanuma oppcommrate="+str(oppcomm_rate)+".mp4", writer="ffmpeg")
  plt.show()
