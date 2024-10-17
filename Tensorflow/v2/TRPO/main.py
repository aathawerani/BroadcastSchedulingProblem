from config import Config
import tensorflow as tf

def main(args=None):

    print(tf.__version__)
    #ng = NetworkGenerator()
    #ng.GenerateAllCases()

    con = Config()
    #con.NetworkStats("network_1")
    #con.NetworkStats("network_2")
    #con.NetworkStats("network_3")
    #con.NetworkStats("network_4")
    #con.NetworkStats("network_5")

    con.network1()
    #con.network2()
    #con.network3()
    #con.network4()
    #con.network5()

    #con.BM1()
    #con.BM2()
    #con.BM3()

    #con.Case1()
    #con.Case6()
    #con.Case11()
    #con.Case16()

    #con.Case2()
    #con.Case7()
    #con.Case12()
    #con.Case17()

    #con.Case3()
    #con.Case8()
    #con.Case13()
    #con.Case18()

    #con.Case4()
    #con.Case9()
    #con.Case14()
    #con.Case19()

    #con.Case5()
    #con.Case10()
    #con.Case15()
    #con.Case20()

if __name__ == "__main__":
    main()

