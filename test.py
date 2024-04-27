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

if __name__ == "__main__":
    words = ["学姐", "漂亮", "有点懒", "很馋"]
    checker = StreamChecker(words=words)
    sent = "漂亮学姐不仅有点懒，还很馋"
    for index, char in enumerate(sent):
        word = checker.query(char)
        if word:
            print(index - len(word) + 1, word)