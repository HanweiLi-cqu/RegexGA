import re
import itertools
from regexNode import *
from random import random, choice
from copy import deepcopy
from math import log
from tqdm import tqdm

OR  = '|'.join # 使用|连接字符
cat = ''.join  # 连接字符
Set = frozenset # 不可改动的set集合

def words(text:str)->set:
    """按照文本中的空格分割文本，返回一个set集合

    Args:
        text (str): 文本

    Returns:
        set: 返回一个set集合
    """
    return set(text.split())


def mistakes(regex:str, M:set, U:set)->set:
    """返回错误的集合，包括winner里未匹配的，loser里匹配的

    Returns:
        set: 错误集合
    """
    return ({"Should have matched: " + W for W in M if not re.search(regex, W)} |
            {"Should not have matched: " + L for L in U if re.search(regex, L)})


def verify(regex:str, M:set, U:set)->bool:
    """验证正则表达式的正确性，如果存在错误集合产生AssertionError,否则返回True

    Args:
        regex (str): 正则表达式
        M (set): winner
        U (set): loser

    Returns:
        bool: 正则表达式是否完全正确
    """
    assert not mistakes(regex, M, U)
    return True


def matches(regex:str, strings:set)->set:
    """返回正则表达式匹配的字符串集合

    Args:
        regex (str): 正则表达式
        strings (set): 字符串集合

    Returns:
        set: 正则表达式匹配的字符串集合
    """
    return {s for s in strings if re.search(regex, s)}


def regex_parts(M:set, U:set)->set:
    """返回一个set集合，包含所有可能的正则表达式集合，这个集合由winner演变，且不匹配loser

    Args:
        M (set): winner
        U (set): loser

    Returns:
        set: 正则表达式集合
    """
    wholes = {'^' + w + '$' for w in M}
    parts = {d for w in wholes for p in subparts(w) for d in dotify(p)}
    return wholes | {p for p in parts if not matches(p, U)}


def subparts(word:str, N:int=4)->set:
    """返回长度不超过N的word:连续字符的子部分集(默认为4)

    Args:
        word (str): 字符串
        N (int, optional): 最长连续长度. Defaults to 4.

    Returns:
        set: 字符串子集
    """
    return set(word[i:i + n + 1] for i in range(len(word)) for n in range(N))


def dotify(part:str)->set:
    """返回字符串集合，这个集合在原有字符串集合的基础上，将可能的字符串替换成dot
    比如 oth {'.t.', 'o.h', 'ot.', 'oth', 'o..', '.th', '...', '..h'}
    Args:
        part (str): 字符串

    Returns:
        set: 字符串集合
    """
    choices = map(replacements, part)
    # 通过itertools.product(*choices)获得所有的可能性（笛卡尔集）
    # 再通过cat进行连接
    return {cat(chars) for chars in itertools.product(*choices)}

def replacements(c:str)->str:
    """在非^和非$的字符后面加上.，否则返回原字符

    Args:
        c (str): 原字符

    Returns:
        str: 替换后的字符
    """
    return c if c in '^$' else c + '.'


def regex_covers(M:set, U:set)->dict:
    """生成正则表达式，并选择不匹配loser的正则表达式和对应的匹配的winner组成的字典

    Args:
        M (set): winner
        U (set): loser

    Returns:
        dict: 结果字典
    """
    losers_str = '\n'.join(U)
    wholes = {'^'+winner+'$' for winner in M} #插入头尾符号 foo -> ^foo$
    parts  = {d for w in wholes for p in subparts(w) for d in dotify(p)} # 获得替换成dot的所有可能性
    reps   = {r for p in parts for r in repetitions(p)} # 在子串集合上再进行处理，加入重复字符串集合
    print(f"wholes:{wholes}")
    print("==============================================")
    print(f"parts:{parts}")
    print("==============================================")
    print(f"reps:{reps}")
    print("==============================================")
    print(f"pairs(M):{pairs(M)}")
    print("==============================================")
    pool   = wholes | parts | pairs(M) | reps # 合并所有的字符串集合,作为可能的正则表达式集合
    searchers = {p:re.compile(p, re.MULTILINE).search for p in pool} #创建字典，key为正则表达式，value为search函数,re.MULTILINE表示多行匹配
    return {p: Set(filter(searchers[p], M))
            for p in pool
            if not searchers[p](losers_str)} #遍历正则表达式集合，如果正则表达式匹配不到loser，就将满足正则表达式的winner加入到字典中，对应的key就是正则表达式


def pairs(winners:set, special_chars=Set('*+?^$.[](){}|\\'))->set:
    """构造新的正则表达式，为winner出现的字母加上重复符号

    Args:
        winners (set): 正确集合
        special_chars (fronzeset, optional): 特殊字符. Defaults to Set('*+?^$.[](){}|\').

    Returns:
        set: 正则表达式集合
    """
    chars = Set(cat(winners)) - special_chars # 获得winner中所有字母，然后去掉特殊符号
    return {A+'.'+q+B
            for A in chars for B in chars for q in '*+?'}#根据字母进行二重循环，然后加入重复符号


def repetitions(part:str)->set:
    """返回通过在每个非特殊字符之后插入单个重复字符('+'或'*'或'?')派生的字符串集。

    Args:
        part (str): 字符串

    Returns:
        set: 加入重复字符的字符串集合
    """
    splits = [(part[:i], part[i:]) for i in range(1, len(part)+1)]
    return {A + q + B
            for (A, B) in splits
            # 不允许产生 '^*' 、 '$*' 、 '..*' 、 '.*.'
            if not (A[-1] in '^$') #不允许在^和$后面加*,+,?这样是没有意义的
            if not A.endswith('..') #不允许在..后面加*,+,?这样是没有意义的
            if not (A.endswith('.') and B.startswith('.')) #不允许在..中间加*,+,?这样是没有意义的
            for q in '*+?'}

def findregex(winners:set, losers:set, k:int=4, addRepetition:bool=False)->str:
    """找到符合条件的正则表达式

    Args:
        winners (set): winner
        losers (set): loser
        k (int, optional): 奖励权重. Defaults to 4.
        addRepetition (bool, optional): 是否考虑重复字符. Defaults to False.

    Returns:
        str: 正则表达式
    """
    if addRepetition:
        # 加入重复字符后
        pool = regex_covers(winners, losers)
    else:
        # 不考虑重复字符
        pool = regex_parts(winners, losers)

    solution = []

    # 评分标准：k为奖励的权重，len(part)为正则表达式的长度
    # 正则表达式匹配的字符串个数乘以奖励权重减去正则表达式的长度
    def score(part):
        return k * len(matches(part, winners)) - len(part)

    # 遍历解决方案，每次选择最佳的正则表达式
    # 最后使用|合并起来，知道winner为空
    while winners:
        best = max(pool, key=score)
        solution.append(best)
        winners = winners - matches(best, winners)
        pool = {r for r in pool if matches(r, winners)}
    return OR(solution)

def scoreFunc(tree, M, U, w=1):
    dif = 0
    regex_str = treeToString(tree)
    M_cn, U_cn = 0, 0
    for s in list(M):
        try:
            if re.search(regex_str, s):
                M_cn += 1
        except Exception as e:
            # print(str(e), "regex_str: ", regex_str)
            return -1
    for u in list(U):
        if re.search(regex_str, u):
            U_cn += 1

    dif = w * (M_cn - 2*U_cn) - len(regex_str)
    return dif

def rankFunc(M, U, population):
    scores = [(scoreFunc(t, M, U), t) for t in population]
    scores_ = []
    for i in scores:
        if i[1]:
            scores_.append(i)
    scores_.sort(key=lambda x:x[0],reverse=True)
    return scores_

def genRandomTree(M, U, charnode_pool,parentnode=rootnode(None,None), splitrate=0.5, concatrate=0.5, charrate=0.5, qualifierate=0.5, maxdepth=12, curren_level=0):
    """随机创建正则表达式树

    Args:
        M (_type_): winner集合
        U (_type_): loser集合
        parentnode (_type_, optional): 父亲节点. Defaults to None.
        splitrate (float, optional): 分割的概率. Defaults to 0.5.
        concatrate (float, optional): 合并节点的概率. Defaults to 0.5.
        charrate (float, optional): 字符节点的概率. Defaults to 0.5.
        qualifierate (float, optional): qualified概率. Defaults to 0.5.
        maxdepth (int, optional): 树的最大深度. Defaults to 12.
        curren_level (int, optional): 当前的level. Defaults to 0.

    Returns:
        _type_: 返回树的节点
    """
    if curren_level > maxdepth:
        return
    # 根节点
    if isinstance(parentnode, rootnode):
        curren_level = 0
        # 根节点的左右子节点为dot占位符号
        rootnode_i = rootnode(
            dotplaceholdernode(None),
            dotplaceholdernode(None)
        )
        # 创建左子节点后面的内容，此时左子节点以及是dot节点了
        rootnode_i.left_child_node = genRandomTree(M, U, charnode_pool,rootnode_i.left_child_node, splitrate, concatrate, charrate,
                                                    qualifierate, maxdepth, curren_level)
        # 创建右子节点后面的内容，此时右子节点以及是dot节点了
        rootnode_i.right_child_node = genRandomTree(M, U, charnode_pool,rootnode_i.right_child_node, splitrate, concatrate, charrate,
                                                     qualifierate, maxdepth, curren_level)
        return rootnode_i

    # (.)占位符号，这个符号后面有三种情况
    # 1.跟着新的分裂符号
    # 2.跟着新的合并符号
    # 3.跟着字符
    if isinstance(parentnode, dotplaceholdernode):
        curren_level += 1
        # "|"分裂符号
        if random() < splitrate:
            return genRandomTree(M, U, charnode_pool,spliternode(None, None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        # ".."合并符号
        elif random() < concatrate:
            return genRandomTree(M, U, charnode_pool,concat_node(None, None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        # "foo"字符
        elif random() < charrate:
            return genRandomTree(M, U, charnode_pool,charnode(None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)

    # "|" 分裂符号，分裂符号后面跟着两个dot占位符号
    if isinstance(parentnode, spliternode):
        curren_level += 1
        splitnode_i = spliternode(
            dotplaceholdernode(None),
            dotplaceholdernode(None)
        )
        splitnode_i.left_childnode = genRandomTree(M, U, charnode_pool,splitnode_i.left_childnode,
                                                               splitrate, concatrate, charrate, qualifierate,
                                                               maxdepth, curren_level)
        splitnode_i.right_childnode = genRandomTree(M, U, charnode_pool,splitnode_i.right_childnode,
                                                                splitrate, concatrate, charrate, qualifierate,
                                                                maxdepth, curren_level)
        return splitnode_i

    # ".." 合并符号，合并符号后面又两种情况
    # 1.跟着字符
    # 2.跟着贪婪量词
    # 3.跟着合并符号
    if isinstance(parentnode, concat_node):
        curren_level += 1
        concat_node_i = concat_node(
            node(None),node(None)
        )
        # 左边节点
        if random() < charrate:
            concat_node_i.left_concatchildnode =  genRandomTree(M, U, charnode_pool,charnode(None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        elif random() < qualifierate:
            concat_node_i.left_concatchildnode =  genRandomTree(M, U, charnode_pool,qualifiernode(None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        else:
            concat_node_i.left_concatchildnode = genRandomTree(M, U, charnode_pool,concat_node(None,None), splitrate, concatrate, charrate,
                                    qualifierate, maxdepth, curren_level)
        # 右边节点
        if random() < charrate:
            concat_node_i.right_concatchildnode =  genRandomTree(M, U, charnode_pool,charnode(None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        # ".+?"
        elif random() < qualifierate:
            concat_node_i.right_concatchildnode =  genRandomTree(M, U, charnode_pool,qualifiernode(None), splitrate, concatrate, charrate,
                                  qualifierate, maxdepth, curren_level)
        else:
            concat_node_i.right_concatchildnode = genRandomTree(M, U, charnode_pool,concat_node(None,None), splitrate, concatrate, charrate,
                                    qualifierate, maxdepth, curren_level)
        return concat_node_i
        

    # 生成字符节点
    if isinstance(parentnode, charnode):
        curren_level += 1
        # 随机选择一个正则表达式，他只从加入定界符和随机替换成dot的字串集合中选取，比如f..
        charnode_str = choice(charnode_pool)
        charnode_i = charnode(charnode_str)
        return charnode_i

    # 生成贪婪量词节点
    if isinstance(parentnode, qualifiernode):
        curren_level += 1
        qualifiernode_str = choice(['.', '+', '?', '*', '.*', '.+', '.*?'])
        qualifiernode_i = qualifiernode(qualifiernode_str)
        return qualifiernode_i

# 基因突变
def mutate(M, U, charnode_pool, t, probchange=0.4):
    if random() < probchange:
        return genRandomTree(M, U, charnode_pool)
    else:
        result = deepcopy(t)
        # 如果节点是cat节点，根据一定的概率决定是否随机用一颗新树代替
        if hasattr(t, "left_concatchildnode"):
            result.left_concatchildnode = mutate(M, U, charnode_pool, t.left_concatchildnode, probchange)
        if hasattr(t, "right_concatchildnode"):
            result.right_concatchildnode = mutate(M, U, charnode_pool, t.right_concatchildnode, probchange)
        # 如果节点是dot节点，则有可能突变
        if hasattr(t, "childnode"):
            result.childnode = mutate(M, U, charnode_pool, t.childnode, probchange)
        # 根据一定的概率决定是否随机从修饰符候选集中选一个新的qualifiernode string代替
        if hasattr(t, "qualifierstrig"):
            result.qualifierstrig = qualifiernode(choice(['.', '+', '?', '*', '.*', '.+', '.*?']))
        # 根据一定的概率决定是否随机从ngram候选列表中选一个新的char sring代替
        if hasattr(t, "charstring"):
            result.charstring = charnode(choice(charnode_pool))

        return result

# 基因交叉
def crossover(t1, t2, probswap=0.7):
    if random() < probswap:
        return deepcopy(t2)
    else:
        result = deepcopy(t1)
        
        if hasattr(t1, 'left_childnode') and hasattr(t2, 'left_childnode'):
            result.left_childnode = crossover(t1.left_childnode, t2.left_childnode, probswap)
        if hasattr(t1, 'right_childnode') and hasattr(t2, 'right_childnode'):
            result.right_childnode = crossover(t1.right_childnode, t2.right_childnode, probswap)
        if hasattr(t1, 'childnode') and hasattr(t2, 'childnode'):
            result.childnode = crossover(t1.childnode, t2.childnode, probswap)
        if hasattr(t1, 'qualifierstrig') and hasattr(t2, 'qualifierstrig'):
            result.qualifierstrig = t2.qualifierstrig
        if hasattr(t1, 'charstring') and hasattr(t2, 'charstring'):
            result.charstring = t2.charstring

    return result 

def evolve(M, U, charnode_pool, popsize=128, rankfunction=rankFunc, maxgen=500, mutationrate=0.6, probswap=0.5, pexp=0.3, pnew=0.8):
    # probexp：表示在构造新种群时，”选择评价较低的程序“这一概率的递减比例。该值越大，相应的筛选过程就越严格，即只选择评价最高的多少比例的个体作为复制对象
    # probexp表示选取的标准，如果probexp很小，那么我们选择的范围就很大
    def selectindex():
        return int(log(random()) / log(pexp))

    # 创建128个初始的种群
    population = [genRandomTree(M, U, charnode_pool) for i in range(popsize)]
    scores = []
    # 产生500代
    for i in tqdm(range(maxgen)):
        # 计算种群得分
        scores = rankfunction(M, U, population)
        # print("evole round: {0}, top score: {1}, regex_str: {2}".format(i, scores[0][0], treeToString(scores[0][1])))
        # 每次取种群的20%进行延续
        # newpop = [scores[0][1], scores[1][1]]
        newpop = [x[1] for x in scores[:int(len(scores)*0.2)]]

        # 产生下一代种群
        # probnew：表示在构造新种群时，”引入一个全新的随机程序“的概率，该参数和probexp是”种群多样性“的重要决定参数
        while len(newpop) < popsize:
            if random() < pnew:
                newpop.append(
                    mutate(
                        M, U, charnode_pool,
                        crossover(
                            scores[selectindex()][1],
                            scores[selectindex()][1],
                            probswap
                        ),
                        mutationrate
                    )
                )
            else:
                # 添加随机的新树
                new_tree = genRandomTree(M, U, charnode_pool)
                newpop.append(new_tree)

        population = newpop
    return scores[0][1]

if __name__ == '__main__':
    M = words('''afoot catfoot dogfoot fanfoot foody foolery foolish fooster footage
        foothot footle footpad footway hotfoot jawfoot mafoo nonfood padfoot prefool sfoot unfool''')

    U = words('''Atlas Aymoro Iberic Mahran Ormazd Silipan altared chandoo crenel crooked
        fardo folksy forest hebamic idgah manlike marly palazzi sixfold tarrock unfold''')
    
    M1 = words('''Mick Rick allocochick backtrick bestick candlestick counterprick 
        heartsick lampwick lick lungsick potstick quick rampick rebrick relick seasick slick tick unsick upstick''')
    
    U1 = words('''Kickapoo Nickneven Rickettsiales billsticker borickite chickell 
        fickleness finickily kilbrickenite lickpenny mispickel quickfoot quickhatch ricksha rollicking slapsticky snickdrawing sunstricken tricklingly unlicked unnickeled''')

    # solution = findregex(M, U, addRepetition=True)
    # if verify(solution, M, U):
    #     print(len(solution), solution)
    
    # exampletree = exampletree()
    # print(exampletree)
    # print(treeToString(exampletree))

    # rd1 = genRandomTree(M, U, parentnode=rootnode(None, None), splitrate=0.5, concatrate=0.5, charrate=0.5,
    #                      qualifierate=0.5, maxdepth=12, curren_level=0)
    # print(rd1)
    # print(treeToString(rd1))
    pattern = input()
    if pattern == "GA":
        charnode_pool = list(regex_parts(M1, U1))
        res = evolve(M1,U1,charnode_pool)
        print(res)
    elif pattern == "N":
        res = findregex(M,U,addRepetition=True)
        print(res)
    else:
        print("Wrong input!")