# Busy Shipper Capstone Project
 Busy Shipper

Problem: A shipper working in a delivery app needs to deliver n packages in a city. Because the workload is too large, he usually has to work overtime. So he decides to write a program to find the shortest route(in meters) to work more effectively. However, there are two important rules that the shipper must obey when working. Firstly, he needs to buy goods before conveying them to customers. Secondly, in the city, he can only handle a given maximum number of packages, if he overcomes this limit, he will receive a punishment (in this case, we assume that he will always follow the rules of the road, even if it can take more time). 

In order to simplify the problem, we will assume that the position of customers and stores are the nodes on the graph, so when we make a test, we will take a list of nodes as an address of customers and stores.

In this topic, we will express a map as a graph, the nodes of a graph are intersections on the map and dead ends. The information will be obtained from “OpenStreetMap”. Because extracting intersections does not belong to our topic, we will use the external library python to get this information.

The program will have some inputs:
1st line: s, m(s is the initial position of the shipper, m is the maximum number of packages that shipper is allowed to handle)
2nd line: n (n packages)
next n lines: <key_address_of_customer> <key_address_of_store>
(We use the Osmnx package to extract the graph, it returns us a graph, each node will have a respective key like the following picture)

The program will have several outputs
+ Time complexity (number of nodes expanded in order to solve the route planning problem)
+ Space complexity (number of nodes kept in memory)
+ The path used to solve the route planning problem (solution) or if there is no possible path between these places, print “Impossible”
+ The cumulated number of meters of the solution (if any)
