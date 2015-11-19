from ai.selection_strategy import RandomStrategy2, SlidingWindow, ImpactBased, NetImpactBased, StrategyFactory
import numpy as np

def get_child_dict(parent_dict):
        child_dict = dict()
        rv_count = len(parent_dict.keys())
        for key in parent_dict.keys():
            child_dict[key] = list()
        for key in parent_dict.keys():
            for val in parent_dict[key]:
                if val < rv_count:
                    child_dict[val].append(key)
                else:
                    child_dict[val - rv_count].append(key + rv_count)
        return child_dict


def test_random_strategy2():
    rs2 = StrategyFactory.generate_selection_strategy('randomStrategy2', pool=range(4), seed=4)
    # rs2 = RandomStrategy2(range(4), 4)
    assert([0, 1] == rs2.choices(2))
    assert(rs2.__str__() == '[rgen seed: 4, pool: [0 1 3 2]]')
    assert([3, 0] == rs2.choices(2))
    assert(rs2.__str__() == '[rgen seed: 4, pool: [3 0 2 1]]')


def test_sliding_window():
    sw = StrategyFactory.generate_selection_strategy('slidingWindow', pool=range(4), seed=4)
    assert(str(sw) == '[rgen seed: 4, pool: [0 1 3 2], deque: [0, 1, 3, 2]]')
    assert((sw.choices(2) == [0, 1]))
    assert(str(sw) == '[rgen seed: 4, pool: [0 1 3 2], deque: [3, 2, 0, 1]]')
    assert(sw.choices(2) == [3, 2])
    assert(str(sw) == '[rgen seed: 4, pool: [0 1 3 2], deque: [0, 1, 3, 2]]')
    assert(sw.choices(2) == [0, 1])
    assert(str(sw) == '[rgen seed: 4, pool: [0 1 3 2], deque: [3, 2, 0, 1]]')

def mock_static_data():
    D = 0
    I = 1
    S = 2
    G = 3
    L = 4
    rvCount = 5
    parentDict = dict()
    parentDict[D] = list()
    parentDict[I] = list()
    parentDict[S] = [I]
    parentDict[G] = [D,I]
    parentDict[L] = [G]
    cpdParams = dict()
    cpdParams[D] = (0, np.zeros((0, 1)), 1)
    cpdParams[I] = (2, np.zeros((0, 1)), 2)
    cpdParams[S] = (15, np.array([[2]]), 1)
    cpdParams[G] = (0, np.array([[-3, 1.5]]), 0.5)
    cpdParams[L] = (0, np.array([[1]]), 0.1)
    child_dict = get_child_dict(parentDict)
    return (rvCount, parentDict, cpdParams, child_dict)


def mock_temporal_data():
    D = 0
    I = 1
    S = 2
    G = 3
    L = 4
    rvCount = 5
    parentDict = dict()
    parentDict[D] = [D + rvCount]
    parentDict[I] = [I + rvCount]
    parentDict[S] = [I, S + rvCount]
    parentDict[G] = [D, I, G + rvCount]
    parentDict[L] = [G, L + rvCount]
    cpdParams = dict()
    cpdParams[D] = [(0, np.zeros((0, 1),dtype=np.float64), 1), (0, np.array([0.75],dtype=np.float64), 1)]
    cpdParams[I] = [(2, np.zeros((0, 1),dtype=np.float64), 2), (0.2, np.array([0.75],dtype=np.float64), 1)]
    cpdParams[S] = [(15, np.array([[2]],dtype=np.float64), 1), (-5, np.array([[1.2]],dtype=np.float64), 1)]
    cpdParams[G] = [(0, np.array([[-3], [1.5]],dtype=np.float64), 0.5), (0, np.array([[-3], [1.5], [1]],
                                                                                     dtype=np.float64), 0.5)]
    cpdParams[L] = [(0, np.array([[1]],dtype=np.float64), 0.1), (0, np.array([[1],[-.3]],dtype=np.float64), 0.1)]
    child_dict = get_child_dict(parentDict)
    return (rvCount, parentDict, cpdParams, child_dict)


def test_impact_based():
    (rvCount, parentDict, cpdParams, child_dict) = mock_temporal_data()
    ib = StrategyFactory.generate_selection_strategy('impactBased', pool=range(5), parentDict=parentDict,
                                                     childDict=child_dict, cpdParams=cpdParams, rvCount=rvCount)
    evidMat = np.array([False] * 5).reshape(-1,1)
    assert(ib.choices(2, 0, evidMat)==[3,1])
    assert(str(ib)=='[pool: [0, 1, 2, 3, 4], rv count: 5]')

def test_net_impact_based():
    (rvCount, parentDict, cpdParams, child_dict) = mock_temporal_data()
    nib = StrategyFactory.generate_selection_strategy('netImpactBased', pool=range(5), parentDict=parentDict,
                                                     childDict=child_dict, cpdParams=cpdParams, rvCount=rvCount)
    evidMat = np.array([False] * 5).reshape(-1,1)
    # print nib.choices(2, 0, evidMat)
    # print nib
    assert(nib.choices(2, 0, evidMat)==[3,1])
    assert(str(nib)=='[pool: [0, 1, 2, 3, 4], rv count: 5]')

# test_random_strategy2()
# test_sliding_window()
# test_impact_based()
test_net_impact_based()
# mock_static_data()
# mock_temporal_data()

# myfunc(bb=98,dd=90)