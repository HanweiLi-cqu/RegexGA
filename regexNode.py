# "ROOT"
class rootnode:
    def __init__(self, left_child_node, right_child_node):
        if left_child_node and right_child_node:
            self.left_child_node = left_child_node
            self.right_child_node = right_child_node
        else:
            self.left_child_node = node
            self.right_child_node = node

    def display(self):
        return "|"
    
    def __str__(self) -> str:
        res = ""
        if self.left_child_node:
            res += self.left_child_node.__str__()
        res += "|"
        if self.right_child_node:
            res += self.right_child_node.__str__()
        return res


# universal child node
class node:
    def __init__(self, node):
        self.node = node

    def __str__(self) -> str:
        return ""


# "|"
class spliternode:
    def __init__(self, left_childnode, right_childnode):
        if left_childnode and right_childnode:
            self.left_childnode = left_childnode
            self.right_childnode = right_childnode
        else:
            self.left_childnode = node
            self.right_childnode = node

    def display(self):
        return "|"
    
    def __str__(self) -> str:
        res = ""
        if self.left_childnode:
            res += self.left_childnode.__str__()
        res += "|"
        if self.right_childnode:
            res += self.right_childnode.__str__()
        return res
        


# "(.)"
class dotplaceholdernode:
    def __init__(self, childnode=None):
        if childnode:
            self.childnode = childnode
        else:
            self.childnode = node
    
    def __str__(self) -> str:
        if self.childnode:
            return self.childnode.__str__()
        else:
            return ""


# "foo"
class charnode:
    def __init__(self, charstring):
        if charstring:
            self.charstring = charstring
        else:
            self.charstring = node

    def display(self):
        return self.charstring
    
    def __str__(self) -> str:
        return self.display()


# ".."
class concat_node:
    def __init__(self, left_concatchildnode, right_concatchildnode):
        if left_concatchildnode and right_concatchildnode:
            self.left_concatchildnode = left_concatchildnode
            self.right_concatchildnode = right_concatchildnode
        else:
            self.left_concatchildnode = node
            self.right_concatchildnode = node
    def __str__(self) -> str:
        res = ""
        if self.left_concatchildnode:
            res += self.left_concatchildnode.__str__()
        if self.right_concatchildnode:
            res += self.right_concatchildnode.__str__()
        return res


# ".+*"贪婪量词
class qualifiernode:
    def __init__(self, qualifierstrig):
        if qualifierstrig:
            self.qualifierstrig = qualifierstrig
        else:
            self.qualifierstrig = node

    def display(self):
        return self.qualifierstrig
    
    def __str__(self) -> str:
        return self.display()

def exampletree():
  return rootnode(
            dotplaceholdernode(
                charnode("foo")
            ),
            dotplaceholdernode(
                concat_node(
                    concat_node(
                        charnode("ba"),
                        qualifiernode("++")
                    ),
                    charnode("r")
                )
            )
        )

# 深度优先遍历打印
def printregextree(rootnode_i):
    if rootnode_i is None:
        return ""

    if isinstance(rootnode_i, rootnode):
        finnal_regexstr = ""
        finnal_regexstr += printregextree(rootnode_i.left_child_node)
        finnal_regexstr += rootnode_i.display()
        finnal_regexstr += printregextree(rootnode_i.right_child_node)
        return finnal_regexstr

    if isinstance(rootnode_i, spliternode):
        split_regexstr = ""
        split_regexstr += printregextree(rootnode_i.left_childnode)
        split_regexstr += rootnode_i.display()
        split_regexstr += printregextree(rootnode_i.right_childnode)
        return split_regexstr

    if isinstance(rootnode_i, dotplaceholdernode):
        return printregextree(rootnode_i.childnode)

    if isinstance(rootnode_i, charnode):
        return rootnode_i.display()

    if isinstance(rootnode_i, concat_node):
        concat_str = ""
        concat_str += printregextree(rootnode_i.left_concatchildnode)
        concat_str += printregextree(rootnode_i.right_concatchildnode)
        return concat_str

    if isinstance(rootnode_i, qualifiernode):
        return rootnode_i.display()