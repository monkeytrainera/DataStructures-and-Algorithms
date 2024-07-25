class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        求word1->word2的最小编辑距离表

        return:
        dp: 最小编辑距离表。dp[i][j]: word1[i-1]->word2[j-1]的最小编辑距离
        """

        l1, l2 = len(word1), len(word2)
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        for i in range(l1+1):
            for j in range(l2+1):
                if i == 0:
                    dp[i][j] = j
                    continue
                elif j == 0:
                    dp[i][j] = i
                    continue
                elif word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp
    
    def backtrackingPath(self, word1: str, word2: str):
        """
        基于最小编辑距离表回溯确定最优编辑操作
        """

        dp = self.minDistance(word1, word2)
        m = len(dp) - 1 # 高，word1
        n = len(dp[0])-1 # 宽，word2
        operations = []

        while n >= 0 or m >= 0:
            print(m, n)
            if n and dp[m][n-1] + 1 == dp[m][n]:
                # 向左，插入word2[n-1]
                operations.append("insert:{}".format(word2[n-1]))
                n -= 1
                continue

            if m and dp[m-1][n] + 1 == dp[m][n]:
                # 向上，删除word1[m-1]
                operations.append("delete:{}".format(word1[m-1]))
                m -= 1
                continue

            if dp[m-1][n-1] + 1 == dp[m][n]:
                # 向左上，word1[m-1]替换成word2[n-1]
                operations.append("replace:{} -> {}".format(word1[m-1], word2[n-1]))
                n -= 1
                m -= 1
                continue

            if dp[m-1][n-1] == dp[m][n]:
                # 向左上，相同，无操作
                operations.append("keep:{}".format(word1[m-1]))
            
            # 减一操作放外面是为了最后能跳出循环，否则可能出现m, n同时为零无法进入最后分支的情况
            n -= 1
            m -= 1
        return operations[::-1]

from typing import *
from collections import deque
class TrieNode():
    def __init__(self):
        self.next = {} # {"a": TrieNode()}
        self.isEnd = False
        self.word = None
        self.fail = None

class Trie():
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        cur = self.root
        for char in word:
            if not cur.next.get(char, None):
                cur.next[char] = TrieNode()
            cur = cur.next[char]
        cur.isEnd = True
        cur.word = word

class StreamChecker:

    def __init__(self, words: List[str]):
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)
        
        q = deque()
        for name, child in self.trie.root.next.items():
            child.fail = self.trie.root
            q.append(child)
        
        while q:
            node = q.popleft()
            node.isEnd = node.isEnd or node.fail.isEnd

            for name, child in node.next.items():
                parent = node
                while parent.fail and not parent.fail.next.get(name, None):
                    parent = parent.fail
                
                child.fail = parent.fail.next[name] if parent.fail else self.trie.root
                q.append(child)

        self.tmp = self.trie.root

    def query(self, letter: str) -> bool:
        if self.tmp.next.get(letter, None):
            self.tmp = self.tmp.next[letter]
            return self.tmp.word
        
        parent = self.tmp
        while parent.fail and not parent.fail.next.get(letter, None):
            parent = parent.fail
        self.tmp = parent.fail.next[letter] if parent.fail else self.trie.root

        return self.tmp.word

# 搜索二叉树
class TreeNode():
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self) -> None:
        self._root = None

    def search(self, num: int) -> bool:
        """查询搜索二叉树中是否存在num
        args:
            num: 需要查询的数字
        
        return:
            bool: 是否存在
        """

        cur = self._root
        while cur is not None:
            if cur.val < num:
                cur = cur.right
            elif cur.val > num:
                cur = cur.left
            else:
                break
        return cur is not None
    
    def insert(self, num: int) -> None:
        """将num插入到搜索二叉树中
        """
        
        if self._root is None:
            self._root = TreeNode(num)
            return
        
        cur, pre = self._root, None
        while cur is not None:
            if cur.val == num:
                return
            
            pre = cur
            if cur.val < num:
                cur = cur.right
            else:
                cur = cur.left
        node = TreeNode(num)
        if pre.val < num:
            pre.right = node
        else:
            pre.left = node
        return
    
    def remove(self, num: int):
        """删除节点"""
        
        # 先找，找不到直接返回
        if self._root is None:
            return
        cur, pre = self._root, None
        while cur is not None:
            if cur.val == num:
                break
            pre = cur
            if cur.val > num:
                cur = cur.left
            else:
                cur = cur.right
        if cur is None:
            return
        
        # 子节点数量小于2，平替即可
        if cur.left is None or cur.right is None:
            child = cur.left or cur.right
            if cur != self._root:
                if pre.left == cur:
                    pre.left = child
                else:
                    pre.right = child
        
            else:
                self._root = child
        
        # 子节点数等于2，找到中序遍历的后继节点，删除后继节点并将后继节点的值覆盖当前节点。
        else:
            tmp = cur.right
            while tmp.left is not None:
                tmp = tmp.left
            
            # 递归删除节点tmp
            self.remove(tmp.val)
            cur.val = tmp.val

            
class TreeNode:
    def __init__(self, val: int) -> None:
        self.val: int = val
        self.height: int = 0
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

class AVL:
    def __init__(self) -> None:
        self._root = None

    def height(self, node: TreeNode) -> int:
        """获取节点的树高"""

        if node is not None:
            return node.height
        return -1
    
    def update_height(self, node: TreeNode):
        """更新节点树高"""

        node.height = max(self.height(node.left), self.height(node.right))

    def balance_factor(self, node: TreeNode) -> int:
        """获取节点平衡因子"""

        if node is None:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def right_rotate(self, node: TreeNode | None) -> TreeNode | None:
        """右旋"""

        child = node.left
        grand_child = child.right

        # 以child为原点，将node右旋
        child.right = node
        node.left = grand_child

        self.update_height(node)
        self.update_height(child)

        return child
    
    def left_rotate(self, node: TreeNode|None) -> TreeNode|None:
        """左旋"""

        child = node.right
        grand_child = child.left

        child.left = node
        node.right = grand_child

        self.update_height(node)
        self.update_height(child)
        
        return child
    
    def rotate(self, node: TreeNode|None) -> TreeNode|None:
        """旋转实现自平衡"""

        balance_factor = self.balance_factor(node)

        # 左偏树
        if balance_factor > 1:
            if self.balance_factor(node.left) >= 0:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        
        # 右偏树
        elif balance_factor < -1:
            if self.balance_factor(node.right) <= 0:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)
        
        # 无偏树 
        return node
    
    def insert_helper(self, node: TreeNode|None, val: int) -> TreeNode:
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self.insert_helper(node.left, val)
        elif val > node.val:
            node.right = self.insert_helper(node.right, val)
        else:
            return node
        
        self.update_height(node)
        return self.rotate(node)
    
    def insert(self, val: int):
        """插入节点"""
        self._root = self.insert_helper(self._root, val)

    def remove_helper(self, node: TreeNode|None, val: int) -> TreeNode|None:
        if node is None:
            return None
        if val < node.val:
            node.left = self.remove_helper(node.left, val)
        elif val > node.val:
            node.right = self.remove_helper(node.right, val)

        else:
            if node.left is None or node.right is None:
                child = node.left or node.right
                if child is None:
                    return None
                else:
                    node = child
            else:
                tmp = node.right
                while tmp.left is not None:
                    tmp = tmp.left
                node.right = self.remove_helper(node.right, val)
                node.val = tmp.val
        self.update_height(node)
        return self.rotate(node)
    
    def remove(self, val: int):
        self._root = self.remove_helper(self._root, val)
    
    def search(self, num: int) -> bool:
        """查询搜索二叉树中是否存在num
        args:
            num: 需要查询的数字
        
        return:
            bool: 是否存在
        """

        cur = self._root
        while cur is not None:
            if cur.val < num:
                cur = cur.right
            elif cur.val > num:
                cur = cur.left
            else:
                break
        return cur is not None
        

        
    


if __name__ == "__main__":
    words = ["学姐", "漂亮", "有点懒", "很馋"]
    checker = StreamChecker(words=words)
    sent = "漂亮学姐不仅有点懒，还很馋"
    for index, char in enumerate(sent):
        word = checker.query(char)
        if word:
            print(index - len(word) + 1, word)