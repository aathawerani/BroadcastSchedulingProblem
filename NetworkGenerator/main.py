import os
import sys
from environmentgenerator import EnvGen
from networkgenerator import NetworkGenerator

def main(args=None):

    ng = NetworkGenerator()
    ng.GenerateAllCases()


if __name__ == "__main__":
    main()

def GenerateNetworkStats():
    casespath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cases')
    print("self.cases", casespath)
    envgen = EnvGen()

    env_name = "network_1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "network_2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "network_3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "network_4"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "network_5"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "BM1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "BM2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "BM3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case1r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case1r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case1r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case2r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case2r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case2r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case3r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case3r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case3r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case4r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case4r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case4r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case5r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case5r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case5r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case6r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case6r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case6r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case7r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case7r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case7r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case8r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case8r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case8r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case9r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case9r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case9r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case10r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case10r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case10r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case11r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case11r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case11r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case12r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case12r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case12r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case13r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case13r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case13r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case14r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case14r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case14r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case15r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case15r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case15r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case16r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case16r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case16r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case17r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case17r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case17r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case18r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case18r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case18r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case19r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case19r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case19r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case20r1"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case20r2"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)

    env_name = "case20r3"
    NumberOfNodes, NumberOfEdges, MaxDegree, MinDegree = envgen.GetNetworkStat(env_name, casespath)
    print(env_name, "NumberOfNodes", NumberOfNodes, "NumberOfEdges", NumberOfEdges, "MaxDegree", MaxDegree, "MinDegree", MinDegree)
