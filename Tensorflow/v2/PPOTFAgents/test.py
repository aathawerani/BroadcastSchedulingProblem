import numpy as np
import sys
import os
from networkgenerator import NetworkGenerator
from netstat import NetworkStats

def SampleNetwork1():
    #grid topology 1
    # 0  1  2
    # 3  4  5
    # 6  7  8

    adj1 = np.zeros((9,9), dtype=np.int16)
    adj1[0][1], adj1[0][3] = 1, 1
    adj1[1][2], adj1[1][4] = 1, 1
    adj1[2][5] = 1
    adj1[3][4], adj1[3][6] = 1, 1
    adj1[4][5], adj1[4][7] = 1, 1
    adj1[5][8] = 1
    adj1[6][7] = 1
    adj1[7][8] = 1
    AddLowerTriangle(adj1)
    AddTwoHop(adj1)
    return adj1

def SampleNetwork2():
    #grid topology 2
    # 0   1   2   3
    # 4   5   6   7
    # 8   9   10  11
    # 12  13  14  15

    adj2 = np.zeros((16,16), dtype=np.int16)
    adj2[0][1], adj2[0][4] = 1, 1
    adj2[1][2], adj2[1][5] = 1, 1
    adj2[2][3], adj2[2][6] = 1, 1
    adj2[3][7] = 1
    adj2[4][5], adj2[4][8] = 1, 1
    adj2[5][6], adj2[5][9] = 1, 1
    adj2[6][7], adj2[6][10] = 1, 1
    adj2[7][11] = 1
    adj2[8][9], adj2[8][12] = 1, 1
    adj2[9][10], adj2[9][13] = 1, 1
    adj2[10][11], adj2[10][14] = 1, 1
    adj2[11][15] = 1
    adj2[12][13] = 1
    adj2[13][14] = 1
    adj2[14][15] = 1
    AddLowerTriangle(adj2)
    AddTwoHop(adj2)
    return adj2

def SampleNetwork3():
    #grid topology 3
    # 0   1   2   3   4
    # 5   6   7   8   9
    # 10 11 12 13 14
    # 15 16 17 18 19
    # 20 21 22 23 24

    adj3 = np.zeros((25,25), dtype=np.int16)
    adj3[0][1], adj3[0][5] = 1, 1
    adj3[1][2], adj3[1][6] = 1, 1
    adj3[2][3], adj3[2][7] = 1, 1
    adj3[3][4], adj3[3][8] = 1, 1
    adj3[4][9] = 1
    adj3[5][6], adj3[5][10] = 1, 1
    adj3[6][7], adj3[6][11] = 1, 1
    adj3[7][8], adj3[7][12] = 1, 1
    adj3[8][9], adj3[8][13] = 1, 1
    adj3[9][14] = 1
    adj3[10][11], adj3[10][15] = 1, 1
    adj3[11][12], adj3[11][16] = 1, 1
    adj3[12][13], adj3[12][17] = 1, 1
    adj3[13][14], adj3[13][18] = 1, 1
    adj3[14][19] = 1
    adj3[15][16], adj3[15][20] = 1, 1
    adj3[16][17], adj3[16][21] = 1, 1
    adj3[17][18], adj3[17][22] = 1, 1
    adj3[18][19], adj3[18][23] = 1, 1
    adj3[19][24] = 1
    adj3[20][21] = 1
    adj3[21][22] = 1
    adj3[22][23] = 1
    adj3[23][24] = 1
    AddLowerTriangle(adj3)
    AddTwoHop(adj3)
    return adj3

def SampleNetwork4():
    #grid topology 4
    # 0   1   2   3   4   5
    # 6   7   8   9   10 11
    # 12 13 14 15 16 17
    # 18 19 20 21 22 23
    # 24 25 26 27 28 29
    # 30 31 32 33 34 35

    adj4 = np.zeros((36,36), dtype=np.int16)
    adj4[0][1], adj4[0][6] = 1, 1
    adj4[1][2], adj4[1][7] = 1, 1
    adj4[2][3], adj4[2][8] = 1, 1
    adj4[3][4], adj4[3][9] = 1, 1
    adj4[4][5], adj4[4][10] = 1, 1
    adj4[5][11] = 1
    adj4[6][7], adj4[6][12] = 1, 1
    adj4[7][8], adj4[7][13] = 1, 1
    adj4[8][9], adj4[8][14] = 1, 1
    adj4[9][10], adj4[9][15] = 1, 1
    adj4[10][11], adj4[10][16] = 1, 1
    adj4[11][17] = 1
    adj4[12][13], adj4[12][18] = 1, 1
    adj4[13][14], adj4[13][19] = 1, 1
    adj4[14][15], adj4[14][20] = 1, 1
    adj4[15][16], adj4[15][21] = 1, 1
    adj4[16][17], adj4[16][22] = 1, 1
    adj4[17][23] = 1
    adj4[18][19], adj4[18][24] = 1, 1
    adj4[19][20], adj4[19][25] = 1, 1
    adj4[20][21], adj4[20][26] = 1, 1
    adj4[21][22], adj4[21][27] = 1, 1
    adj4[22][23], adj4[22][28] = 1, 1
    adj4[23][29] = 1
    adj4[24][25], adj4[24][30] = 1, 1
    adj4[25][26], adj4[25][31] = 1, 1
    adj4[26][27], adj4[26][32] = 1, 1
    adj4[27][28], adj4[27][33] = 1, 1
    adj4[28][29], adj4[28][34] = 1, 1
    adj4[29][35] = 1
    adj4[30][31] = 1
    adj4[31][32] = 1
    adj4[32][33] = 1
    adj4[33][34] = 1
    adj4[34][35] = 1
    AddLowerTriangle(adj4)
    AddTwoHop(adj4)
    return adj4

def SampleNetwork5():
    #grid topology 5
    # 0   1  2  3  4  5  6
    # 7   8  9 10 11 12 13
    # 14 15 16 17 18 19 20
    # 21 22 23 24 25 26 27
    # 28 29 30 31 32 33 34
    # 35 36 37 38 39 40 41
    # 42 43 44 45 46 47 48

    adj5 = np.zeros((49,49), dtype=np.int16)
    adj5[0][1], adj5[0][7] = 1, 1
    adj5[1][2], adj5[1][8] = 1, 1
    adj5[2][3], adj5[2][9] = 1, 1
    adj5[3][4], adj5[3][10] = 1, 1
    adj5[4][5], adj5[4][11] = 1, 1
    adj5[5][6], adj5[5][12] = 1, 1
    adj5[6][13] = 1
    adj5[7][8], adj5[7][14] = 1, 1
    adj5[8][9], adj5[8][15] = 1, 1
    adj5[9][10], adj5[9][16] = 1, 1
    adj5[10][11], adj5[10][17] = 1, 1
    adj5[11][12], adj5[11][18] = 1, 1
    adj5[12][13], adj5[12][19] = 1, 1
    adj5[13][20] = 1
    adj5[14][15], adj5[14][21] = 1, 1
    adj5[15][16], adj5[15][22] = 1, 1
    adj5[16][17], adj5[16][23] = 1, 1
    adj5[17][18], adj5[17][24] = 1, 1
    adj5[18][19], adj5[18][25] = 1, 1
    adj5[19][20], adj5[19][26] = 1, 1
    adj5[20][27] = 1
    adj5[21][22], adj5[21][28] = 1, 1
    adj5[22][23], adj5[22][29] = 1, 1
    adj5[23][24], adj5[23][30] = 1, 1
    adj5[24][25], adj5[24][31] = 1, 1
    adj5[25][26], adj5[25][32] = 1, 1
    adj5[26][27], adj5[26][33] = 1, 1
    adj5[27][34] = 1
    adj5[28][29], adj5[28][35] = 1, 1
    adj5[29][30], adj5[29][36] = 1, 1
    adj5[30][31], adj5[30][37] = 1, 1
    adj5[31][32], adj5[31][38] = 1, 1
    adj5[32][33], adj5[32][39] = 1, 1
    adj5[33][34], adj5[33][40] = 1, 1
    adj5[34][41] = 1
    adj5[35][36], adj5[35][42] = 1, 1
    adj5[36][37], adj5[36][43] = 1, 1
    adj5[37][38], adj5[37][44] = 1, 1
    adj5[38][39], adj5[38][45] = 1, 1
    adj5[39][40], adj5[39][46] = 1, 1
    adj5[40][41], adj5[40][47] = 1, 1
    adj5[41][48] = 1
    adj5[42][43] = 1
    adj5[43][44] = 1
    adj5[44][45] = 1
    adj5[45][46] = 1
    adj5[46][47] = 1
    adj5[47][48] = 1
    AddLowerTriangle(adj5)
    AddTwoHop(adj5)
    return adj5

def GetOneHop(adj, index):
    onehop = []
    for j in range(len(adj)):
        if adj[index][j] == 1:
            onehop.append(j)
    return onehop

def AddTwoHop(adj):
    for i in range(len(adj)):
        onehop = GetOneHop(adj, i)
        #print("onehop", onehop)
        twohop = []
        for node in onehop:
            twohop.append(GetOneHop(adj, node))
        #print("twohop", twohop)
        for nodelist in twohop:
            for node in nodelist:
                if i != node and adj[i][node] == 0:
                    adj[i][node] = 2
                #print("i", i, "node", node, "adj[i][node]", adj[i][node])

def AddLowerTriangle(adj):
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0:
                adj[j][i] = adj[i][j]

def BM1():
    adj = np.zeros((15,15), dtype=np.int16)
    adj[1-1][2-1], adj[1-1][3-1] = 1, 1
    adj[2-1][4-1], adj[2-1][5-1] = 1, 1
    adj[3-1][5-1], adj[3-1][6-1], adj[3-1][7-1] = 1, 1, 1
    adj[4-1][5-1], adj[4-1][8-1], adj[4-1][9-1] = 1, 1, 1
    adj[5-1][6-1], adj[5-1][9-1], adj[5-1][10-1] = 1, 1, 1
    adj[6-1][10-1], adj[6-1][11-1], adj[6-1][12-1] = 1, 1, 1
    adj[7-1][12-1] = 1
    adj[8-1][9-1] = 1
    adj[9-1][10-1], adj[9-1][13-1] = 1, 1
    adj[10-1][11-1], adj[10-1][13-1], adj[10-1][14-1], adj[10-1][15-1] = 1, 1, 1, 1
    adj[11-1][12-1], adj[11-1][15-1] = 1, 1
    adj[12-1][15-1] = 1
    adj[13-1][14-1] = 1
    adj[14-1][15-1] = 1
    AddLowerTriangle(adj)
    AddTwoHop(adj)
    return adj

def BM2():
    adj = np.zeros((30,30), dtype=np.int16)
    adj[1-1][2-1], adj[1-1][7-1], adj[1-1][8-1], adj[1-1][10-1], adj[1-1][25-1] = 1, 1, 1, 1, 1
    adj[2-1][8-1], adj[2-1][9-1], adj[2-1][11-1], adj[2-1][26-1] = 1, 1, 1, 1
    adj[3-1][5-1], adj[3-1][6-1], adj[3-1][27-1] = 1, 1, 1
    adj[4-1][7-1], adj[4-1][9-1], adj[4-1][11-1], adj[4-1][19-1], adj[4-1][21-1], adj[4-1][23-1], adj[4-1][28-1] = 1, 1, 1, 1, 1, 1, 1
    adj[5-1][10-1], adj[5-1][29-1]  = 1, 1
    adj[6-1][8-1], adj[6-1][12-1], adj[6-1][13-1], adj[6-1][30-1] = 1, 1, 1, 1
    adj[7-1][11-1], adj[7-1][14-1] = 1, 1
    adj[8-1][10-1], adj[8-1][13-1] = 1, 1
    adj[10-1][12-1], adj[10-1][13-1], adj[10-1][14-1], adj[10-1][20-1], adj[10-1][21-1] = 1, 1, 1, 1, 1
    adj[11-1][15-1] = 1
    adj[12-1][15-1], adj[12-1][16-1], adj[12-1][19-1], adj[12-1][22-1] = 1, 1, 1, 1
    adj[13-1][16-1] = 1
    adj[14-1][17-1], adj[14-1][18-1], adj[14-1][22-1], adj[14-1][25-1], adj[14-1][27-1] = 1, 1, 1, 1, 1
    adj[15-1][18-1] = 1
    adj[16-1][18-1], adj[16-1][20-1]  = 1, 1
    adj[17-1][19-1] = 1
    adj[18-1][19-1], adj[18-1][22-1] = 1, 1
    adj[19-1][20-1], adj[19-1][21-1], adj[19-1][22-1] = 1, 1, 1
    adj[20-1][22-1], adj[20-1][27-1], adj[20-1][29-1] = 1, 1, 1
    adj[21-1][23-1], adj[21-1][24-1] = 1, 1
    adj[22-1][24-1] = 1
    adj[23-1][24-1] = 1
    adj[24-1][26-1] = 1
    adj[25-1][29-1], adj[25-1][27-1] = 1, 1
    adj[26-1][30-1], adj[26-1][28-1] = 1, 1
    adj[27-1][28-1], adj[27-1][30-1] = 1, 1
    adj[28-1][29-1] = 1
    adj[29-1][30-1] = 1
    AddLowerTriangle(adj)
    AddTwoHop(adj)
    return adj

def BM3():
    adj = np.zeros((40,40), dtype=np.int16)
    adj[1-1][2-1], adj[1-1][9-1] = 1, 1
    adj[2-1][3-1], adj[2-1][10-1] = 1, 1
    adj[3-1][4-1], adj[3-1][11-1] = 1, 1
    adj[4-1][5-1], adj[4-1][12-1] = 1, 1
    adj[5-1][6-1], adj[5-1][13-1] = 1, 1
    adj[6-1][15-1] = 1
    adj[7-1][8-1], adj[7-1][14-1] = 1, 1
    adj[8-1][16-1] = 1
    adj[9-1][10-1] = 1
    adj[10-1][11-1], adj[10-1][17-1] = 1, 1
    adj[11-1][19-1], adj[11-1][20-1] = 1, 1
    adj[12-1][13-1], adj[12-1][19-1] = 1, 1
    adj[13-1][14-1], adj[13-1][20], adj[13-1][21-1] = 1, 1, 1
    adj[14-1][15-1], adj[14-1][22-1] = 1, 1
    adj[15-1][23-1] = 1
    adj[16-1][23-1], adj[16-1][24-1] = 1, 1
    adj[17-1][25-1] = 1
    adj[18-1][19-1], adj[18-1][25-1] = 1, 1
    adj[19-1][20-1], adj[19-1][26-1], adj[19-1][27-1], adj[19-1][28-1] = 1, 1, 1, 1
    adj[20-1][21-1], adj[20-1][28-1] = 1, 1
    adj[21-1][22-1], adj[21-1][30-1] = 1, 1
    adj[22-1][23-1], adj[22-1][29-1], adj[22-1][31-1] = 1, 1, 1
    adj[23-1][24-1], adj[23-1][31-1] = 1, 1
    adj[24-1][31-1], adj[24-1][32-1] = 1, 1
    adj[25-1][33-1] = 1
    adj[26-1][27-1], adj[26-1][33-1] = 1, 1
    adj[27-1][28-1], adj[27-1][35-1] = 1, 1
    adj[28-1][29-1], adj[28-1][35-1], adj[28-1][36-1] = 1, 1, 1
    adj[29-1][38-1] = 1
    adj[30-1][31-1], adj[30-1][38-1] = 1, 1
    adj[31-1][38-1] = 1
    adj[32-1][40-1] = 1
    adj[34-1][35-1] = 1
    adj[36-1][37-1] = 1
    adj[37-1][38-1] = 1
    adj[38-1][39-1] = 1
    adj[39-1][40-1] = 1
    AddLowerTriangle(adj)
    AddTwoHop(adj)
    return adj

np.set_printoptions(threshold=sys.maxsize)

#print("SampleNetwork1")
#adjacent = SampleNetwork1()
#print(adjacent)

#print("SampleNetwork2")
#adjacent = SampleNetwork2()
#print(adjacent)

#print("SampleNetwork3")
#adjacent = SampleNetwork3()
#print(adjacent)

#print("SampleNetwork4")
#adjacent = SampleNetwork4()
#print(adjacent)

#print("SampleNetwork5")
#adjacent = SampleNetwork5()
#print(adjacent)

#print("BM1")
#adjacent = BM1()
#print(adjacent)

#print("BM2")
#adjacent = BM2()
#print(adjacent)

#print("BM3")
#adjacent = BM3()
#print(adjacent)

def GetNetwork(name):
    cases = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cases')
    ng = NetworkGenerator()
    adjacent = ng.ReadNetwork2(name, cases)
    #AddLowerTriangle(adjacent)
    #AddTwoHop(adjacent)
    #ns = NetworkStats(casespath)
    #NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree, nodeedges = ns.CalculateStats2(env_name, adjacent)
    #print(name)
    #print(adjacent)

    NumberOfEdges = 0
    NumberOfNodes = len(adjacent)
    nodeedges = np.zeros((NumberOfNodes))
    for a in range(NumberOfNodes):
        for b in range(NumberOfNodes):
            if adjacent[a][b] == 1:
                NumberOfEdges += 1
                nodeedges[a] += 1
                nodeedges[b] += 1
    MaxDegree = np.max(nodeedges)
    MinDegree = np.min(nodeedges)

    print("case", name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges,
        "MaxDegree", MaxDegree, "MinDegree", MinDegree)


#GetNetwork("case1r1")
#GetNetwork("case1r2")
#GetNetwork("case1r3")
#GetNetwork("case2r1")
#GetNetwork("case2r2")
#GetNetwork("case2r3")
#GetNetwork("case3r1")
#GetNetwork("case3r2")
#GetNetwork("case3r3")
#GetNetwork("case4r1")
#GetNetwork("case4r2")
#GetNetwork("case4r3")
#GetNetwork("case5r1")
#GetNetwork("case5r2")
#GetNetwork("case5r3")

#GetNetwork("case6r1")
#GetNetwork("case6r2")
#GetNetwork("case6r3")
#GetNetwork("case11r1")
#GetNetwork("case11r2")
#GetNetwork("case11r3")
#GetNetwork("case16r1")
#GetNetwork("case16r2")
#GetNetwork("case16r3")
#GetNetwork("case7r1")
#GetNetwork("case7r2")
#GetNetwork("case7r3")
#GetNetwork("case12r1")
#GetNetwork("case12r2")
#GetNetwork("case12r3")
#GetNetwork("case17r1")
#GetNetwork("case17r2")
#GetNetwork("case17r3")
#GetNetwork("case8r1")
#GetNetwork("case8r2")
#GetNetwork("case8r3")
#GetNetwork("case13r1")
#GetNetwork("case13r2")
#GetNetwork("case13r3")
#GetNetwork("case18r1")
#GetNetwork("case18r2")
#GetNetwork("case18r3")
#GetNetwork("case9r1")
#GetNetwork("case9r2")
#GetNetwork("case9r3")
#GetNetwork("case14r1")
#GetNetwork("case14r2")
#GetNetwork("case14r3")
#GetNetwork("case19r1")
#GetNetwork("case19r2")
#GetNetwork("case19r3")
#GetNetwork("case10r1")
#GetNetwork("case10r2")
#GetNetwork("case10r3")
#GetNetwork("case15r1")
#GetNetwork("case15r2")
#GetNetwork("case15r3")
#GetNetwork("case20r1")
#GetNetwork("case20r2")
#GetNetwork("case20r3")

class Node():
    def __init__(self, i, j):
        self.i = i
        self.j = j

def GenerateRandomNetwork(N, r):
    nodes = []

    for k in range(N):
        l = np.random.uniform(0, 1)
        m = np.random.uniform(0, 1)
        n = Node(l, m)
        nodes.append(n)

    #for n in nodes:
        #print("n.i", n.i, "n.j", n.j)

    #print("start edge generation")
    adj = np.zeros((N, N), dtype=np.int16)
    for i in range(N):
        for j in range(i + 1, N):
            xi = nodes[i].i
            yi = nodes[i].j
            xj = nodes[j].i
            yj = nodes[j].j
            dist = np.sqrt(np.square(xi - xj) + np.square(yi - yj))
            if dist <= r:
                adj[i][j] = 1
            #print("xi", xi, "yi", yi, "xj", xj, "yj", yj, "dist", dist, "r", r, "i", i, "j", j, "adj[i][j]", adj[i][j])
    # print(adj)
    return adj

def WriteNetwork2(name, adj):
    filename = "D:\\aaht14\\PhD\\IBA\\WIP\\BSP\\Code20230722\\New folder\\" + name + ".txt"
    f1 = open(filename, 'a')
    length = len(adj)
    for i in range(length):
        for j in range(length):
            element = str(adj[i][j]) + ", "
            f1.write(element)
        f1.write("\n")

def GenerateCase(case, N, r):
    print("start case", case, "N", N, "r", r)
    adjacent = GenerateRandomNetwork(N, r)
    WriteNetwork2(case, adjacent)
    print("end case", case, "N", N, "r", r)

    NumberOfEdges = 0
    NumberOfNodes = len(adjacent)
    nodeedges = np.zeros((NumberOfNodes))
    for a in range(NumberOfNodes):
        for b in range(NumberOfNodes):
            if adjacent[a][b] == 1:
                NumberOfEdges += 1
                nodeedges[a] += 1
                nodeedges[b] += 1
    MaxDegree = np.max(nodeedges)
    MinDegree = np.min(nodeedges)

    if(MinDegree == 0):
        for a in range(NumberOfNodes):
            for b in range(NumberOfNodes):
                if nodeedges[a] == 0 and nodeedges[b] == 0 :
                    adjacent[a][b] == 1
                    NumberOfEdges += 1
                    nodeedges[a] += 1
                    nodeedges[b] += 1
    MaxDegree = np.max(nodeedges)
    MinDegree = np.min(nodeedges)


    print("case", case, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges,
        "MaxDegree", MaxDegree, "MinDegree", MinDegree)

GenerateCase("case1r1", 100, 1 / np.sqrt(100))
GenerateCase("case1r2", 100, 1 / np.sqrt(100))
GenerateCase("case1r3", 100, 1 / np.sqrt(100))
GenerateCase("case2r1", 300, 1 / np.sqrt(300))
GenerateCase("case2r2", 300, 1 / np.sqrt(300))
GenerateCase("case2r3", 300, 1 / np.sqrt(300))
GenerateCase("case3r1", 500, 1 / np.sqrt(500))
GenerateCase("case3r2", 500, 1 / np.sqrt(500))
GenerateCase("case3r3", 500, 1 / np.sqrt(500))
GenerateCase("case4r1", 750, 1 / np.sqrt(750))
GenerateCase("case4r2", 750, 1 / np.sqrt(750))
GenerateCase("case4r3", 750, 1 / np.sqrt(750))
GenerateCase("case5r1", 1000, 1 / np.sqrt(1000))
GenerateCase("case5r2", 1000, 1 / np.sqrt(1000))
GenerateCase("case5r3", 1000, 1 / np.sqrt(1000))
