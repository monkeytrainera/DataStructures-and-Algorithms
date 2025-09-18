- [时间复杂度](#时间复杂度)
- [排序](#排序)
  - [选择排序](#选择排序)
  - [冒泡排序](#冒泡排序)
  - [补充知识：异或运算](#补充知识异或运算)
  - [插入排序](#插入排序)
  - [补充知识：二分法](#补充知识二分法)
  - [补充知识：Master公式](#补充知识master公式)
  - [归并排序](#归并排序)
  - [快排](#快排)
  - [堆排](#堆排)
  - [桶排序](#桶排序)
  - [排序算法总结](#排序算法总结)
- [链表](#链表)
  - [反转链表](#反转链表)
  - [打印升序链表公共部分](#打印升序链表公共部分)
  - [判断一个链表是否是回文结构](#判断一个链表是否是回文结构)
  - [将单向链表按某值划分成左边小、中间相等、右边大的形式](#将单向链表按某值划分成左边小中间相等右边大的形式)
  - [复制含有随机指针的链表](#复制含有随机指针的链表)
  - [链表相交问题](#链表相交问题)
- [二叉树](#二叉树)
  - [二叉树的遍历](#二叉树的遍历)
  - [二叉树的Morris遍历](#二叉树的morris遍历)
  - [二叉树相关概念](#二叉树相关概念)
  - [树型DP方法](#树型dp方法)
  - [寻找公共祖先节点](#寻找公共祖先节点)
  - [后继节点](#后继节点)
  - [二叉树的序列化和反序列化](#二叉树的序列化和反序列化)
  - [折纸问题](#折纸问题)
- [图](#图)
  - [图克隆](#图克隆)
  - [拓扑排序（课程表问题）](#拓扑排序课程表问题)
  - [补充知识：并查集（Disjoint Set Union）](#补充知识并查集disjoint-set-union)
  - [最小生成树](#最小生成树)
    - [Kruskal算法](#kruskal算法)
    - [Prim 算法](#prim-算法)
  - [Dijkstra算法](#dijkstra算法)
- [前缀树和贪心算法](#前缀树和贪心算法)
- [暴力递归](#暴力递归)
  - [汉诺塔问题](#汉诺塔问题)
  - [字符串的全部子序列](#字符串的全部子序列)
  - [字符串的全排列](#字符串的全排列)
  - [预测赢家问题](#预测赢家问题)
  - [解码问题](#解码问题)
  - [背包问题](#背包问题)
  - [N皇后问题](#n皇后问题)
- [哈希函数和哈希表](#哈希函数和哈希表)
  - [哈希函数](#哈希函数)
  - [哈希表的实现](#哈希表的实现)
  - [布隆过滤器](#布隆过滤器)
  - [一致性哈希](#一致性哈希)
- [KMP字符串匹配](#kmp字符串匹配)
- [Manacher算法](#manacher算法)
- [滑动窗口最值](#滑动窗口最值)
- [单调栈](#单调栈)
- [动态规划](#动态规划)
  - [引例](#引例)
  - [预测赢家问题](#预测赢家问题-1)
  - [下象棋问题](#下象棋问题)
  - [零钱兑换](#零钱兑换)
  - [编辑距离](#编辑距离)
  - [有效括号](#有效括号)
- [有序表](#有序表)
  - [搜索二叉树实现](#搜索二叉树实现)
  - [AVL树实现](#avl树实现)
- [AC自动机](#ac自动机)
  - [字符流问题](#字符流问题)



# 时间复杂度

常数操作：如果一个操作消耗的时间是固定的，与数据量没关系（如数组寻址），则该操作为常数操作。

时间复杂度就是衡量一个算法流程中常数操作数量的指标。

首先根据算法流程确定常数操作数量的表达式，然后取最高阶项作为时间复杂度。

评价一个算法的好坏，先看时间复杂度，如果相同就比较不同数据样本下的实际运行时间，即“常数项时间”

# 排序

## 选择排序

$i\sim N-1$遍历，每次选出最小值放在$i$位置。时间复杂度$O(N^2)$，空间复杂度$O(1)$.

## 冒泡排序

每次遍历时，将$arr[i]与arr[i+1]$比较，大值右移。这样，第$i$次遍历一次就确定第$i$大的值，遍历$N-1$次即可。时间复杂度$O(N^2)$，空间复杂度$O(1)$。

## 补充知识：异或运算

异或运算：同0异1

异或运算可以理解成二进制下的无进位相加。

性质：

- $0\bigoplus{N}=N,N\bigoplus{N}=0$
- 满足交换律和结合律

应用：

- 两数交换：$a=a\bigoplus{b},b=a\bigoplus{b},a=a\bigoplus{b}$
- 数组中只有一种数出现过奇数次，怎么找出这个数：从头到尾异或一遍即可
- 数组中只有两种数出现过奇数次，找出这两个数：先从头到尾异或得到$eor=a\bigoplus{b}$，将$eor$取反加一与运算得到$rightone$(保留$eor$最右边的一个1，其余位置零)。假设$rightone$的二进制第$k$位是1，说明两数中有一个的第$k$位是1，另一个是0。因此，将所有二进制第$k$位是1的数(跟$rightone$做与运算判断)从头到尾异或一遍就可以得到其中一个数，再将这个数与$eor$异或就可以得到另一个数。

## 插入排序

每次操作使得$0\sim i$位置上的数有序，$i=1\sim N-1$操作$N-1$次即可。对于每次操作，$0\sim i-1$位置已经有序，将$i$位置的数与前面的数倒序依次比较，顺序不对则交换，顺序正确则该次操作完成，即$0\sim i$位置上已有序。

插入排序的时间复杂度与数据情况关系很大。比如顺序排列12345，只需要比较4次，时间复杂度是$O(N)$，而顺序排列54321，时间复杂度就是$O(N^2)$。

确定算法的时间复杂度时，考虑最差情况。所以插入排序的算法复杂度为$O(N^2)$，空间复杂度是$O(1)$

## 补充知识：二分法

二分法的时间复杂度是$O(log_2N)$，比如数组中一共8个数，二分3次。一般简写成$O(logN)$.

例题：

1. 有序数组判断某数是否存在：二分，根据中点判断。
2. 顺序数组找≥某个数最左侧的位置：二分，如果中点满足≥的条件就用一个变量$ans$记录该数的位置，然后取左半部分继续二分，并不断更新$ans$；否则，取右半部分继续二分。二分结束后（只剩下一个数）得到的最终的$ans$就是最左侧的位置。与第一题的不同在于这次一定要二分到底才能确定答案，
3. 无序数组局部最小值：给定一个相邻元素不相等的无序数组，找出任意一个局部最小值：首先判断首尾是否存在局部最小值（$arr[0]<arr[1] 或者 arr[N-1]>arr[N]$）。如果不存在，则说明$arr[0]和arr[N]$中间一定存在最小值。取中点$arr[mid]$，先判断$arr[mid]$是否为局部最小值，如果不是，$arr[mid]>arr[mid-1]$取左半部分继续二分，否则取右半部分继续二分。

## 补充知识：Master公式

$$
T(N)=a*T(\frac{N}{b})+O(N^d)
$$

N是母问题的数据量，$\frac{N}{b}$是子问题的数据量，$O(N^d)$是除了子问题以外其余操作的时间复杂度

符合这种模式的递归算法的复杂度可以根据Master公式直接得到：
$$
\left\{
\begin{align}
&O(N^d),&log_ba<d\\
&O(N^{log_ba}),&log_ba>d\\
&O(N^d*logN),&log_ba=d
\end{align}
\right.
$$

## 归并排序

采用递归的方式排序，二分为左右两部分，各自排好序后再合并到一起。

例题：

1. 数组升序排列

   ```python
   class Solution(object):
       def sortArray(self, nums):
           """
           :type nums: List[int]
           :rtype: List[int]
           """
   
           def Merge(left, mid, right):
               i, j = left, mid + 1
               res = []
               while i <= mid and j <= right:
                   if nums[i] < nums[j]:
                       res.append(nums[i])
                       i += 1
                   else:
                       res.append(nums[j])
                       j += 1
               if i > mid:
                   res += nums[j : right + 1]
               else:
                   res += nums[i : mid + 1]
               for i in range(len(res)):
                   nums[left + i] = res[i]
   
           def mergeSort(left, right):
               if left == right:
                   return
               mid = left + ((right - left) >> 1)
               mergeSort(left, mid)
               mergeSort(mid + 1, right)
               Merge(left, mid, right)
   
           if len(nums) < 2:
               return nums
           mergeSort(0, len(nums) - 1)
           return nums
   ```

   复杂度分析：符合Master公式，$a=b=2,d=1$，因此复杂度为$O(NlogN)$

2. 小和问题：在一个数组中，每一个数左边比当前数小的数的和加起来，称为该数组的小和

   力扣315：给你一个整数数组 $nums$，按要求返回一个新数组$counts$。数组$counts$有该性质： $counts[i]$ 的值是  $nums[i]$右侧小于$nums[i]$的元素的数量。

   解题思路：归并排序的Merge中进行计数

   代码：

   ```python
   class Solution(object):
       def countSmaller(self, nums):
           """
           :type nums: List[int]
           :rtype: List[int]
           """
   
           def Merge(left, mid, right):
               i, j = left, mid + 1
               res = []
               while i <= mid and j <= right:
                   if nums[i][0] > nums[j][0]:
                       count[nums[i][1]] += right - j + 1
                       res.append(nums[i])
                       i += 1
                   else:
                       res.append(nums[j])
                       j += 1
               if i > mid:
                   res += nums[j : right + 1]
               else:
                   res += nums[i : mid + 1]
               for i in range(len(res)):
                   nums[left + i] = res[i]
   
           def mergeSort(left, right):
               if left == right:
                   return
               mid = left + ((right - left) >> 1)
               mergeSort(left, mid)
               mergeSort(mid + 1, right)
               Merge(left, mid, right)
   
           count = [0 for _ in nums]
           nums = [(nums[i], i) for i in range(len(nums))] #[数值，序号]
           if len(nums) < 2:
               return [0]
           mergeSort(0, len(nums) - 1)
           return count
   ```

   时间复杂度同例1，空间复杂度为$O(N)$，Merge中会用一个辅助数组，其长度最大为$N$。

3. 逆序对问题：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

   思路：与上题类似，Merge中计数即可。

   代码：

   ```python
   class Solution(object):
       def reversePairs(self, nums):
           """
           :type nums: List[int]
           :rtype: int
           """
           
           def Merge(left, mid, right):
               i, j = left, mid + 1
               res = []
               while i <= mid and j <= right:
                   if nums[i] > nums[j]:
                       self.count += right - j + 1
                       res.append(nums[i])
                       i += 1
                   else:
                       res.append(nums[j])
                       j += 1
               if i > mid:
                   res += nums[j : right + 1]
               else:
                   res += nums[i : mid + 1]
               for i in range(len(res)):
                   nums[left + i] = res[i]
   
           def mergeSort(left, right):
               if left == right:
                   return
               mid = left + ((right - left) >> 1)
               mergeSort(left, mid)
               mergeSort(mid + 1, right)
               Merge(left, mid, right)
   
           self.count = 0
           if len(nums) < 2:
               return 0
           mergeSort(0, len(nums) - 1)
           return self.count
   ```

   复杂度分析同例2。疑问：空间复杂度需不需要考虑压栈的空间复杂度。

   归并排序思考：每排好一部分，都将这部分变成一个整体参与之后的操作。这样可以既简化了后续的排序操作（merge即可），也保证了在merge操作中进行的统计能实现不重不漏。

## 快排

引例：荷兰国旗问题。给定一个数组和一个数，将数组中小于该数的放左边，等于的放中间，大于的放右边。要求时间复杂度$O(N)$，空间复杂度$O(1)$

设置两个变量：

- left：“小于”区域的右边界，即$arr[:left+1]$都小于给定数
- right：“大于”区域的左边界，即$arr[right:]$都大于给定数

假设给定数为$a$，从头遍历整个数组，假设当前位置为$i$：

- 如果$arr[i]<a$，将当前数与左边界后一位交换，左边界右移一位，当前数跳下一个（$i=i+1$）
- 如果$arr[i]==a$,当前数跳下一个（$i=i+1$）
- 如果$arr[i]<a$，将当前数与右边界前一位交换，右边界左移一位

快排：对于一个数组，取最右边的数作为给定数$a$，剩下的根据荷兰国旗问题进行分段。记录下"<"部分的右边界和">"部分的左边界，然后对两段继续重复该操作。

时间复杂度$O(N^2)$，空间复杂度$O(N)$。最坏情况，每次最右边的数都是剩下的数组中最大的，那么每次都是只能确定一个数的位置。

快排改进版：

最坏情况出现的原因是划分值（给定数$a$）太偏。如果每次都能正好打在中间，则可以应用Master公式，求出时间复杂度为$O(NlogN)$，这是最好的情况。这时的空间复杂度为$O(logN)$。

改进方法：每次划分时，从划分序列中**随机选取一个元素**与最后的元素做交换，然后同上。这样可以保证划分值时序列中随机选取的，假设序列为$[1,8,2,2,2,2,4]$，那么随机选到2的可能性最大，这样一次划分操作就可以确定四个2的位置，大大提高效率。数学上可以证明改进版的时间复杂度是$O(NlogN)$

代码：

```python
import random
class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def partition(start, end):
            p = random.randint(start, end)
            nums[p], nums[end] = nums[end], nums[p]
            pivot = nums[end]
            left, right = start - 1, end
            i = start
            while i < right:
                if nums[i] < pivot:
                    nums[i], nums[left + 1] = nums[left + 1], nums[i]
                    left += 1
                    i += 1
                elif nums[i] == pivot:
                    i += 1
                else:
                    nums[i], nums[right - 1] = nums[right - 1], nums[i]
                    right -= 1
            nums[right], nums[end] = nums[end], nums[right]

            return left, right + 1

        def quickSort(start, end):
            if end <= start:
                return
            left, right = partition(start, end)
            quickSort(start, left)
            quickSort(right, end)
        
        quickSort(0, len(nums) - 1)
        return nums                 

```

## 堆排

堆结构很重要。

堆结构就是完全二叉树结构。数组中从零开始连续的一段，可以写成完全二叉树的结构。对应关系：数组中$i$位置的元素，对应到完全二叉树中节点，其左节点在数组中的位置为$2*i+1$，右节点为$2*i+2$，父节点为$(i-1)>>1$，堆结构节点的个数用heapsize表示。

大根堆：每棵子树的头节点都是该子树的最大值

小根堆：每棵子树的头节点都是该子树的最小值

堆结构的两个基本操作（以小根堆为例）：

1. heapinsert：对于当前节点，如果该节点比自己的父节点小，就交换自己跟父节点的位置（同时新的父节点继续往上跟父节点比较）

   ```python
           def heapinsert(nums, index):
               '''
               nums: 数组
               index: 元素下标
               '''
               while ((index - 1) >> 1) >= 0:
                   if nums[index] > nums[(index - 1) >> 1]:
                       nums[index], nums[(index - 1) >> 1] = nums[(index - 1) >> 1], nums[index]
                       index = (index - 1) >> 1
                   else:
                       break
   ```

2. heapify：从根节点开始，将其与较小的子节点比较，如果根节点小则结束，否则根节点与较小子节点交换，然后以较小子节点作为新的根节点继续与其较小子节点比较

   ```python
           def heapify(nums, index, heapsize):
               '''
               大根堆
               heapsize:堆元素个数
               '''
               left = 2 * index + 1
               while left < heapsize:
                   if left + 1 < heapsize and nums[left] < nums[left + 1]:
                       large = left + 1
                   else:
                       large = left
                   if nums[index] >= nums[large]:
                       break
                   else:
                       nums[index], nums[large] = nums[large], nums[index]
                       index = large
                       left = 2 * index + 1
   ```

   两个操作的时间复杂度都是$O(logN)$。因为完全二叉树的高度就是$logN$级别。

应用：

1. 堆中任意一个位置的数变了，如何调整使得堆结构仍成立：在改变的位置，先heapinsert，再heapify（两个操作只有一个会工作）。

2. 给定一个数组，变成堆结构：

   - 从头开始heapinsert一遍，操作N次。时间复杂度$O(NlogN)$。

   - 从$arr[\frac{N}{2}-1]$往前heapify一遍。时间复杂度$O(N)$。

     证明：

     假设是满完全二叉树，N个数据，则有子节点的最下面一层是$\frac{N}{4}$级别，往上依次为$\frac{N}{8},\frac{N}{16},\frac{N}{32}...$。常数操作依次为2,3,4,...

     由此可写出时间复杂度为
     $$
     T(N)=\frac{N}{4}+\frac{N}{8}*2+\frac{N}{16}*3+\frac{N}{32}*4+...
     $$
     上式两边×2，得
     $$
     2T(N)=\frac{N}{2}+\frac{N}{4}*2+\frac{N}{8}*3+\frac{N}{16}*4+...
     $$
     错位相减，得
     $$
     \begin{align}
     T(N)&=\frac{N}{2}+\frac{N}{4}+\frac{N}{8}+...\\
     &=N\frac{\frac{1}{2}*(1-(\frac{1}{2})^n)}{1-\frac{1}{2}}\\
     &=N(1-(\frac{1}{2})^n)
     \end{align}
     $$
     时间复杂度为$O(N)$

3. 堆排序：

   ```python
           #先将数组变成大根堆
           for i in range(len(nums) / 2 - 1, -1, -1):
               heapify(nums, i, len(nums))
           #每次固定一个最大值，heapsize减一，然后堆根做heapify
           heapsize = len(nums)
           while heapsize > 1:
               nums[0], nums[heapsize - 1] = nums[heapsize - 1], nums[0]
               heapsize -= 1
               heapify(nums, 0, heapsize)
           return nums
   ```

4. 堆排序拓展：几乎有序的数组排序，要求时间复杂度低于$O(NlogN)$（几乎有序是指：如果将数组排好顺序，每个元素移动的距离不超过K，并且K相对于数组来说比较小）

   首先实现一个堆结构，或者直接调用现成的

   ```python
   class large_root_heap(object):
       
       def __init__(self, arr=None):
           self.arr = []
           self.heapsize = 0
           if arr:
               for _ in arr:
                   self.arr.append(_)
                   self.heapsize += 1
               for i in range((len(self.arr) >> 1) - 1, -1, -1):
                   self.heapify(i)
                   
       def heapify(self, index):
           left = 2 * index + 1
           while left < self.heapsize:
               if left + 1 < self.heapsize and self.arr[left] < self.arr[left + 1]:
                   large = left + 1
               else:
                   large = left
               if self.arr[index] >= self.arr[large]:
                   break
               else:
                   self.arr[large], self.arr[index] = self.arr[index], self.arr[large]
                   index = large
                   left = 2 * index + 1
                   
       def heapinsert(self, index):
           while ((index - 1) >> 1) >= 0 and self.arr[index] > self.arr[(index - 1) >> 1]:
               self.arr[index], self.arr[(index - 1) >> 1] = self.arr[(index - 1) >> 1], self.arr[index]
               index = (index - 1) >> 1
           
       def add(self, value):
           self.arr.append(value)
           self.heapinsert(self.heapsize)
           self.heapsize += 1
   
       def pop(self):
           self.arr[0], self.arr[self.heapsize - 1] = self.arr[self.heapsize - 1], self.arr[0]
           res = self.arr.pop()
           self.heapsize -= 1
           self.heapify(0)
           return res
   ```
   
   首先将数组前K个数放入堆，然后维持堆的大小向后滑动到数组尾。
   
   ```python
   def Kheapsort(k, arr):
       heap = large_root_heap()
       for i in range(k):
           heap.add(arr[i])
       index = 0
       i += 1
       while i < len(arr):
           arr[index] = heap.pop()
           index += 1
           heap.add(arr[i])
           i += 1
       while index < len(arr):
           arr[index] = heap.pop()
           index += 1   
       print(arr)
   ```
   
   这样时间复杂度是$O(NlogK)$。

## 桶排序

前面所有的排序都是基于比较的排序。桶排序是基于计数的排序。

计数排序：基于计数的排序适用范围比较小，比如员工年龄排序，年龄区间不大，适合基于计数的排序，时间复杂度是$O(N)$。

基数排序：准备十个桶（队列），先按个位数放入队列，然后0~9桶依次把数字倒出（先进先出），然后按十位数，直到最高位。

基数排序的实现：

```python
        def getDigit(num, d):
        '''
        取某一位上的数字（d=0代表个位）
        '''
            if len(str(num)) <= d:
                return 0
            return int(str(num)[-(d + 1)])

        def radixSort(nums, digit):
            radix = 10
            # 辅助数组
            bucket = [0 for _ in nums]
            # 最大数有digit位，要进出桶digit次
            for d in range(digit):
                # 初始化一个数组，用于统计
                count = [0 for _ in range(radix)]
                # 统计数组中数字的第d位上的数字
                for num in nums:
                    j = getDigit(num, d)
                    count[j] += 1
                # 将统计数组改成前缀和数组
                for i in range(1, radix):
                    count[i] = count[i - 1] + count[i]
                # 将原数组中的数按照第d位上的数字以及统计数组的信息放入辅助数组
                for i in range(len(nums) - 1, -1, -1):
                    j = getDigit(nums[i], d)
                    bucket[count[j] - 1] = nums[i]
                    count[j] -= 1
                # 辅助数组中的数倒回到原数组，本次进出桶结束
                for i in range(len(bucket)):
                    nums[i] = bucket[i]
        
        max_num = max(nums)
        digit = len(str(max_num))
        radixSort(nums, digit)
```

（没太懂，以后再看）

## 排序算法总结

排序稳定性。以学生排序为例，假设学生有年龄和体重两个属性，先按照年龄排序，然后按照体重排序，如果第二次排序是稳定的，则相同的体重的学生的年龄是有序的。

| 排序算法 | 时间复杂度 | 空间复杂度 | 稳定性 |
| -------- | ---------- | ---------- | ------ |
| 选择     | $O(N^2)$   | $O(1)$     | 0      |
| 冒泡     | $O(N^2)$   | $O(1)$     | 1      |
| 插入     | $O(N^2)$   | $O(1)$     | 1      |
| 归并     | $O(NlogN)$ | $O(N)$     | 1      |
| 随机快排 | $O(NlogN)$ | $O(logN)$  | 0      |
| 堆       | $O(NlogN)$ | $O(1)$     | 0      |

分析：

- 后三者时间复杂度都比较低。其中，归并的优势是具有稳定性，劣势是空间复杂度较高；随机快排的优势是常数时间小，实际排序速度最快，劣势是空间复杂度较大，且不具备稳定性；堆排的优势是空间复杂度很小，劣势是不具备稳定性。
- 基于计数的排序具备稳定性。

补充：组合排序，结合$O(N^2)和O(NlogN)$算法各自的优势。大样本上用快排，当样本足够小时改用插入排序。插入排序在小样本上速度更快。

# 链表

## 反转链表

时间复杂度$O(N)$，空间复杂度$O(1)$​。给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。（力扣92）

```python
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummyNode = ListNode(0)
        dummyNode.next = head
        pre = dummyNode

        for _ in range(left - 1):
            pre = pre.next
        
        cur = pre.next
        for _ in range(right - left):
            next = cur.next
            cur.next = next.next
            next.next = pre.next
            pre.next = next
        return dummyNode.next
```



## 打印升序链表公共部分

两个指针分别指向两个链表的head，谁小谁向后移动，相等打印并共同向后移动，有一个越界则停止

## 判断一个链表是否是回文结构

要求时间复杂度$O(N)$，空间复杂度$O(1)$。

假设空间复杂度没有要求，可直接考虑压栈（优化一下可以只前一半压栈）。但是空间复杂度有要求，就要考虑后半部分链表倒序。

```python
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        def reverse(head):
            if not head or not head.next:
                return head
            pre, cur, next = None, head, head.next
            while next:
                cur.next = pre
                pre = cur
                cur = next
                next = next.next
            cur.next = pre
            return cur

        # 快慢指针确定中点
        fast, slow = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # 将右半部分倒叙
        if fast.next:
            head2 = slow.next
            slow.next = None
            head2 = reverse(head2)
        else:
            head2 = reverse(slow)

        # 判断回文
        while head:
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next
        return True
```

## 将单向链表按某值划分成左边小、中间相等、右边大的形式

如果没有其他要求，直接放在数组里，转化成荷兰国旗问题。

如有以下要求：时间复杂度$O(N)$，空间复杂度$O(1)$，具有稳定性。则需要利用三组首尾指针。

```python
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        # 三段头节点
        head1, head2, head3 = None, None, None
        # 遍历链表
        p = head
        while p:
            if p.val < x:
                if head1:
                    p1.next = p
                    p1 = p1.next
                else:
                    head1 = p
                    p1 = head1
            elif p.val == x:
                if head2:
                    p2.next = p
                    p2 = p2.next
                else:
                    head2 = p
                    p2 = head2            
            else:
                if head3:
                    p3.next = p
                    p3 = p3.next
                else:
                    head3 = p
                    p3 = head3
            p = p.next
        # 串联三段
        res = ListNode(0)
        p = res
        if head1:
            p.next = head1
            p = p1
        if head2:
            p.next = head2
            p = p2
        if head3:
            p.next = head3
            p = p3
        p.next = None
        return res.next
```

## 复制含有随机指针的链表

如果空间复杂度没有要求，可以直接定义一个链表节点的字典，键-值对为老节点-新节点对，然后根据老链表将新链表的指针确定好即可。

如果要求不用字典，直接在原链表上操作，即要求空间复杂度是$O(1)$

```python
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return
        # 先在每个节点后复制出一个新节点
        p = head
        while p:
            newNode = Node(p.val)
            newNode.next = p.next
            p.next = newNode
            p = p.next.next
        
        # 确定新节点随机指针
        p = head
        while p:
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        
        # 新旧链表分离
        p1, p2 = head, head.next
        res = p2
        while p1.next.next:
            p1.next = p2.next
            p1 = p1.next
            p2.next = p1.next
            p2 = p2.next
        p1.next = p2.next
        p2.next = None
        return res
```

## 链表相交问题

给定两个可能有环也可能无环的单链表，判断两链表是否相交，如果相交返回相交的第一个节点。假设两个链表长度和为N，要求时间复杂度为$O(N)$，空间复杂度为$O(1)$。

分析：

- 两个链表如果相交，只可能都有环或者都无环
- 如果无环，求出两链表长度差，从头部开始，长链表指针先走长度差步，然后长短链表的指针一起走，重合位置即为相交的第一个节点

- 如果有环，有三种可能：两链表各自有环，不相交；入环前相交（两链表入环节点相同）；入环后相交（两链表入环节点不同）

判断链表是否有环：

```python
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 单个节点肯定无环
        if not head or not head.next:
            return
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        # 没有第一次相遇肯定无环
        if fast != slow:
            return 
        # 第二次相遇确定入环节点
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return fast
```

无环链表判断相交节点：

```python
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB:
            return 
        # 求出两链表长度
        p1, p2 = headA, headB
        n = 0
        while p1.next:
            n += 1
            p1 = p1.next
        while p2.next:
            n -= 1
            p2 = p2.next
        # 最后一个不同肯定不相交
        if p1 != p2:
            return
        # 长链表指针先走
        p1, p2 = headA, headB
        if n < 0:
            plong, pshort = headB, headA
        else:
            plong, pshort = headA, headB
        for _ in range(abs(n)):
            plong = plong.next
        # 判断相交节点
        while plong:
            if plong == pshort:
                return plong
            plong = plong.next
            pshort = pshort.next
        return 
```

有环链表判断相交节点：

判断情况2：看入环节点是否相同。将入环节点看成终点，就变成了无环问题。

判断情况1、3：将第一个链表从入环节点继续往下走，在回到自己之前能遇见第二个链表的入环节点即为情况3，相交节点为任意一个入环节点。否则是情况1，即不相交。

# 二叉树

## 二叉树的遍历

dfs遍历

递归法：dfs遍历二叉树，每一个节点会过三次

```python
        def dfs(root):
            # 1
            if not root:
                return
            # 1 
            dfs(root.left)
            # 2
            # 2
            dfs(root.right)
            # 3
            # 3
```

先序遍历在1处打印，中序遍历在2处打印，后序遍历在3处打印。

迭代法：

先序遍历：

```python
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return
        stack = [root]
        res = []
        while stack:
            root = stack.pop()
            res.append(root.val)
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
        return res
```

后序遍历：先序遍历先进左子节点再进右子节点，最后结果倒序即可。

中序遍历：

```python
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
```

bfs遍历：队列

二叉树层序遍历：要求每层节点在一个数组里

```python
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            temp = []
            for _ in range(len(stack)):
                root = stack.pop(0)
                temp.append(root.val)
                if root.left:
                    stack.append(root.left)
                if root.right:
                    stack.append(root.right)
            res.append(temp)
        return res
```

## 二叉树的Morris遍历

时间复杂度为$O(N)$，空间复杂度为$O(1)$。利用原树结构中大量的空闲指针来节省空间。

遍历细节：

假设来到当前节点cur（开始时cur在头节点位置）

1. 如果当前cur没有左孩子，cur向右孩子移动
2. 如果当前cur有左孩子，找到左子树的最右节点mostRight：
   - 如果mostRight的右指针指向空，将其指向当前节点cur，然后cur左移
   - 如果mostRight的右指针指向cur（这说明是第二次来到cur），将其指向空，然后cur右移

3. cur为空时遍历停止

代码实现：

```python
        def Morris(root):
            if not root:
                return
            while root:
                mostRight = root.left
                if mostRight:   
                    # 有左子树
                    while mostRight.right and mostRight.right != root: # 找到左子树的最右节点
                        mostRight = mostRight.right
                    if not mostRight.right:   # 最右节点为空，第一次来到cur
                        mostRight.right = root  
                        root = root.left
                        continue
                    else:                     # 最右节点指向当前节点，第二次来到
                        mostRight.right = None
                root = root.right
```

Morris遍历中每一个有左孩子的cur节点会被遍历两次，第一次会建立左子树mostRight到cur的指针，第二次通过指针从mostRight再次回到cur。根据Morris遍历可以很方便的改出以下遍历方式：

1. 先序遍历：

   ```python
       def preorderTraversal(self, root):
           """
           :type root: TreeNode
           :rtype: List[int]
           """
           def Morris(root):
               if not root:
                   return
               while root:
                   mostRight = root.left
                   if mostRight:   
                       # 有左子树
                       while mostRight.right and mostRight.right != root: # 找到左子树的最右节点
                           mostRight = mostRight.right
                       if not mostRight.right:   # 最右节点为空，第一次来到cur
                           res.append(root.val)  # 有左孩子，第一遍历次打印
                           mostRight.right = root  
                           root = root.left
                           continue
                       else:                     # 最右节点指向当前节点，第二次来到
                           mostRight.right = None
                   else:
                       res.append(root.val)      # 没有左孩子，直接打印
                   root = root.right  
   
           res = []
           Morris(root)
           return res
   ```

2. 中序遍历：

   ```python
       def inorderTraversal(self, root):
           """
           :type root: TreeNode
           :rtype: List[int]
           """
           def Morris(root):
               if not root:
                   return
               while root:
                   if root.left:
                       mostRight = root.left
                       while mostRight.right and mostRight.right != root:
                           mostRight = mostRight.right
                       if not mostRight.right:
                           mostRight.right = root
                           root = root.left
                           continue # 这里保证第一次来到当前节点时不会打印
                       else:
                           mostRight.right = None
                   res.append(root.val) 
                   root = root.right 
           
           res = []
           Morris(root)
           return res
   ```

3. 后序遍历：

   ```python
       def postorderTraversal(self, root):
           """
           :type root: TreeNode
           :rtype: List[int]
           """
           def reverseEdge(root):
               '''
               将树最右节点的路径看成单链表即可，单链表逆序问题
               '''
               pre = None
               while root:
                   next = root.right
                   root.right = pre
                   pre = root
                   root = next
               return pre  # 返回逆序后的头节点
   
           def printEdge(root):
               '''
               先逆序，再打印，最后再逆序恢复原状
               '''
               tail = reverseEdge(root) 
               cur = tail
               while cur:
                   res.append(cur.val)
                   cur = cur.right
               reverseEdge(tail)
   
           def Morris(root):
               if not root:
                   return
               while root:
                   if root.left:
                       mostRight = root.left
                       while mostRight.right and mostRight.right != root:
                           mostRight = mostRight.right
                       if not mostRight.right:
                           mostRight.right = root
                           root = root.left
                           continue
                       else:
                           mostRight.right = None
                           printEdge(root.left) # 第二次来到cur节点时打印其子树最右节点的逆序
                   root = root.right
           
           head = root # 记录根节点
           res = []
           Morris(root)
           printEdge(head) # 最后打印整棵树的最右节点的逆序
           return res
   ```

   

## 二叉树相关概念 

搜索二叉树：

- 左子树节点一定都小于根节点
- 右子树节点一定大于根节点
- 左右子树都是搜索二叉树

判断是否为搜索二叉树：中序遍历、递归

递归写法如下：

bfs:

```python
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def bfs(lower, upper, root):
            if not root:
                return True
            if root.val <= lower or root.val >= upper:
                return False
            return bfs(lower, root.val, root.left) and bfs(root.val, upper, root.right)
        return bfs(float('-inf'), float('inf'), root)
```

dfs:

```python
        def dfs(root):
            if not root:
                return True
            if not dfs(root.left):
                return False
            # 在中序遍历打印的位置进行比较操作
            if root.val <= self.prevalue:
                return False
            else:
                self.prevalue = root.val
            return dfs(root.right)
        self.prevalue = float('-inf')
        return dfs(root)
```

Morris遍历：

```python
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def Morris(root):
            if not root:
                return True
            pre = float('-inf')
            while root:
                if root.left:
                    mostRight = root.left
                    while mostRight.right and mostRight.right != root:
                        mostRight = mostRight.right
                    if not mostRight.right:
                        mostRight.right = root
                        root = root.left
                        continue
                    else:
                        mostRight.right = None
                if root.val <= pre:
                    return False
                pre = root.val
                root = root.right
            return True
        return Morris(root)
```

搜索二叉树实现：
```

```

完全二叉树：任何一个节点，如果有右子节点，一定有左子节点。

完全二叉树判断：层序遍历

- 如果任意一个节点有右子节点无左子节点，false
- 上条不违反的话，第一个左右子节点不全的节点后面全为叶节点

```python
    def isCompleteTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        stack = [root]
        leaf = False
        while stack:
            root = stack.pop(0)
            if (leaf and (root.left or root.right)) or (root.right and not root.left):
                return False
            if root.left:
                stack.append(root.left)
            if root.right:
                stack.append(root.right)
            if not (root.left and root.right):
                leaf = True
        return True
```

满二叉树：每一层都满的完全二叉树。假设树高为$h$，节点数为$n$，则满足$n=2^h-1$

平衡二叉树：任意节点两子树高度差不超过1.

## 树型DP方法

后两种树结构的判断都可以建模成**树型DP问题：根据左右子树返回的信息体，确定当前树的信息体返回，实现递归。问题的难点在于信息体的确定。**判断满二叉树问题的信息体可以为树高和节点数。根节点的信息体即为整棵树的树高和节点数，然后根据公式判断即可。判断平衡二叉树问题的信息体可以为树高和是否为平衡二叉树。

以平衡二叉树判断为例：这里将信息体中的树高和是否平衡两个信息合并了

```python
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def check(root):
            if not root:
                return 0
            left = check(root.left)
            right = check(root.right)
            if left < 0 or right < 0 or abs(left - right) > 1:
                return -1
            return max(left, right) + 1
        res = check(root)
        return res >= 0
```

树型DP方法应用举例：

1. 二叉树最大路径和：

   ```python
       def maxPathSum(self, root):
           """
           :type root: TreeNode
           :rtype: int
           """
           def maxpath(root):
               if not root:
                   return 0
               left = maxpath(root.left)
               right = maxpath(root.right)
               self.ans = max(self.ans, left + root.val + right)  # 更新最大距离
               return max(0, max(left, right) + root.val)         # 返回树高  
           self.ans = float('-inf')
           maxpath(root)
           return self.ans
   ```

2. 派对的最大快乐值：https://www.nowcoder.com/questionTerminal/a5f542742fe24181b28f7d5b82e2e49a

   整个公司的人员结构可以看作是一棵标准的多叉树。树的头节点是公司唯一的老板，除老板外，每个员工都有唯一的直接上级，叶节点是没有任何下属的基层员工，除基层员工外，每个员工都有一个或多个直接下级，另外每个员工都有一个快乐值。 

     这个公司现在要办 party，你可以决定哪些员工来，哪些员工不来。但是要遵循如下的原则： 

     1.如果某个员工来了，那么这个员工的所有直接下级都不能来。 

     2.派对的整体快乐值是所有到场员工快乐值的累加。 

     3.你的目标是让派对的整体快乐值尽量大。 

     给定一棵多叉树，请输出派对的最大快乐值。

   ```python
   # 处理输入
   import sys
   n, root = map(int, sys.stdin.readline().split()) # 员工数量， 根节点
   arr = [[] for i in range(n)]                     # 存储树结构
   values = map(int, sys.stdin.readline().split())  # 存储每个员工的快乐值
   while True:                                      # 接收输入并构建树结构
       line = map(int, sys.stdin.readline().split())
       if not line:
           break
       arr[line[0] - 1].append(line[1] - 1)
   root = root - 1                             
   
   def process(root):
       if not arr[root]:               # 叶节点结果可直接返回
           return (values[root], 0)
       laimax, bumax = values[root], 0 # 初始化当前来与不来的两种情况
       for node in arr[root]:
           laimax_node, bumax_node = process(node) # 拿到子节点来与不来的最大值
           laimax += bumax_node                    # 当前节点来，子节点就都不来
           bumax += max(laimax_node, bumax_node)   # 当前节点不来，取子节点来与不来的较大值
       return (laimax, bumax)                      # 构造当前节点的返回值
   print(max(process(root)))
   ```

   

## 寻找公共祖先节点

给定二叉树中两个节点node1和node2，找出他们的最低公共祖先节点。

简单思路：生成一个字典，记录所有的子节点-父节点对。先根据字典确定node1父节点集合set1（包含node1），然后从node2开始向上遍历父节点，第一个在set1中出现的节点即为最低公共祖先节点

高级玩法：递归

只有两种情况：最低公共子节点为p or q；最低公共子节点是其他节点

```python
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # 遇见空节点或者p、q节点就往上抛
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果左右子节点都不为空，把根节点往上抛（对应第二种情况）
        if left and right:
            return root
        # 把不为空的节点（肯定是p or q）往上抛
        if left:
            return left
        return right
```

## 后继节点

后继节点：中序遍历结果中，每一个节点的后一个节点。对应的，前一个节点为前驱节点。

假如普通二叉树结构，需要中序遍历，时间复杂度$O(N)$。

现给定特殊二叉树，每一个节点都有一个指针指向自己的父节点，要求确定后继节点，时间复杂度小于$O(N)$。

分析：假设当前节点为x

- 如果x有右子节点，则后继节点是右子树的最左节点
- 如果x无右子节点，沿着父节点指针往上看，看x是否为某个父节点的左子树的节点。如果不是，继续向上判断；如果是，则该父节点即为后继节点(x是该父节点左子树的最右节点) 。如果找不到该父节点，则x是整个二叉树最右节点，其后继节点为空

```python
    def getSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        def getMostLeft(root):
            while root.left:
                root = root.left
            return root

        if not root:
            return
        if root.right:
            return getMostLeft(root.right)
        else:
            parent = root.parent
            while parent and parent.left != root:
                root = parent
                parent = root.parent
            return parent
```

## 二叉树的序列化和反序列化

简单点说，就是内存里的一棵树如何变成字符串形式，以及如何由字符串形式变回内存里的一棵树

官方定义：序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

bfs先序遍历的方式：

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return '#_'
        res = str(root.val) + '_'
        res += self.serialize(root.left)
        res += self.serialize(root.right)
        return res
    
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        datalst = data.split('_')
        def process():
            value = datalst.pop(0)
            if value == '#':
                return 
            root = TreeNode(int(value))
            root.left = process()
            root.right = process()
            return root
        return process()
```

bfs层序遍历的方式：

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        stack = [root]
        res = ''
        while stack:
            root = stack.pop(0)
            if not root:
                res += '#_'
            else:
                res += str(root.val) + '_'
                stack.append(root.left)
                stack.append(root.right)
        return res
    
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def process(value):
            if value == '#':
                return 
            else:
                return TreeNode(int(value))

        datalst = data.split('_')
        res = process(datalst.pop(0))
        if not res:
            return 
        stack = [res]
        while stack:
            root = stack.pop(0)
            root.left = process(datalst.pop(0))
            root.right = process(datalst.pop(0))
            if root.left:
                stack.append(root.left)
            if root.right:
                stack.append(root.right)
        return res
```

## 折纸问题

把纸条竖着放在桌⼦上，然后从纸条的下边向上⽅对折，压出折痕后再展 开。此时有1条折痕，突起的⽅向指向纸条的背⾯，这条折痕叫做“下”折痕   ；突起的⽅向指向纸条正⾯的折痕叫做“上”折痕。如果每次都从下边向上⽅ 对折，对折N次。请从上到下计算出所有折痕的⽅向。 

给定折的次数**n**,请返回从上到下的折痕的数组，若为下折痕则对应元素为"down",若为上折痕则为"up"。

分析：本质上是一个N层的完全二叉树结构。整棵树的根节点是'down'，任意节点的左子节点都是'down'，右子节点都是'up'。最终从上到下的数组即为中序遍历的结果。终止条件是层数，也就是折纸次数。

```python
class FoldPaper:
    def foldPaper(self, n):
        def process(i, res, sign):
            # i代表当前在第几层，防止越界
            if i > n:
                return
            process(i + 1, res, 'down')
            # 之前的打印操作
            res.append(sign)
            process(i + 1, res, 'up')
        res = []
        process(1, res, 'down')
        return res
```

# 图

表达图的方法很多，常见的有邻接表和邻接矩阵。

一个很详细的表达图信息的模板，对于极复杂的问题可以用该模板对图信息进行整理

```python
class Node(object):
    def __init__(self, value):
        self.value = value   # 节点的数据项，多为int/string
        self.in_num = 0      # 节点的入度
        self.out_num = 0     # 节点的出度
        self.nexts = set()   # 节点发散出去的边连接的相邻节点
        self.edges = set()   # 节点发散出去的边

class Edge(object):
    def __init__(self, weight, from_node, to_node):
        self.weight = weight        # 边的权值
        self.from_node = from_node  # 边的头节点
        self.to_node = to_node      # 边的尾节点
        
class Graph(object):
    def __init__(self):
        self.nodes = dict()  # 图的点集{点编号：点对象}
        self.edges = set()   # 图的边集
```

## 图克隆

给你无向连通图中一个节点的引用，请你返回该图的深拷贝（克隆），图的表示方法是邻接表。

dfs（递归）:

```python
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        seen = {}
        def dfs(node):
            # 判断node是否为空
            if not node:
                return node
            # 判断node是否见过
            if node in seen:
                return seen[node]
            # 克隆当前node，并将其字典中的value设置成克隆node
            # 注意加入seen的操作一定要放在递归之前，否则递归会无限循环下去
            # 要先将当前node放入seen，再递归的处理neighbors
            clone_node = Node(node.val)
            seen[node] = clone_node
            # 递归处理neighbors
            if node.neighbors:
                clone_node.neighbors = [dfs(n) for n in node.neighbors]
            return clone_node
        return dfs(node)
```

bfs:

```python
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node
        res = node
        stack = [node]
        seen = {node : Node(node.val)}
        # bfs
        while stack:
            node = stack.pop(0)
            for n in node.neighbors:
                if n not in seen:
                    stack.append(n)
                    seen[n] = Node(n.val)
                # 当前node的neighbors克隆
                seen[node].neighbors.append(seen[n])
        return seen[res]
```

dfs（非递归）：

```python
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        # dfs（非递归）
        while stack:
            node = stack.pop()
            for n in node.neighbors:
                if n not in seen:
                    stack.append(node)
                    stack.append(n)
                    seen[n] = Node(n.val)
                    break
                # 注意dfs时每个node可能多次进栈，所以处理neighbors要判断是否重复，这点不同于bfs
                if seen[n] not in seen[node].neighbors:
                    seen[node].neighbors.append(seen[n])
        return seen[res]
```

## 拓扑排序（课程表问题）

判断有向无环图。

在选修某些课程之前需要一些先修课程。先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi]，表示如果要学习课程ai则必须先学习课程bi。请你判断是否可能完成所有课程的学习？

bfs：队列 + 入度表 （很经典，很重要）

```python
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        #bfs
        # 根据元素数据对构建入度表和邻接表
        indegree = [0 for _ in range(numCourses)]
        nexts = [[] for _ in range(numCourses)]
        for lst in prerequisites:
            indegree[lst[0]] += 1
            nexts[lst[1]].append(lst[0])
        # 将入度为0的点放入队列
        stack = []
        for i in range(numCourses):
            if indegree[i] == 0:
                stack.append(i)
        # 用队列实现bfs
        while stack:
            i = stack.pop(0)
            numCourses -= 1
            for node in nexts[i]:
                indegree[node] -= 1
                if indegree[node] == 0:
                    stack.append(node)
        # 如果是有向无环图，则stack归零时，numCourses也归零
        return not numCourses
```

dfs：遍历节点判断节点是否在环中

```python
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # dfs
        # 如果有环，则一定存在一个节点dfs时会回到自己，即一定有一个节点在环中
        # 检验所有节点都不在环中即为无环图
        def dfs(i):
            if flags[i] == -1:  # -1 代表该节点检验合格，不在环中
                return True
            if flags[i] == 1:   # 1 代表该节点在dfs中被2次访问，说明该节点在环中
                return False
            flags[i] = 1        # 将当前节点置1，代表第一次访问
            for j in nexts[i]:  # dfs检查当前节点是否在环中（会不会2次访问）
                if not dfs(j):
                    return False
            flags[i] = -1       # 检验合格后，将该节点置-1，标记为不在环中
            return True
        # 标记数组，初始值为0    
        flags = [0 for _ in range(numCourses)] 
        # 构建邻接表
        nexts = [[] for _ in range(numCourses)]
        for lst in prerequisites:
            nexts[lst[1]].append(lst[0])
        # dfs 遍历每一个节点，判断其是否在环中
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True
```

## 补充知识：并查集（Disjoint Set Union）

主要用于解决**元素分组**问题，管理一系列**不相交的集合**，支持两种操作：

- Union（合并）：将两个不相交的集合合并成一个集合
- Find（查询）：查询两个元素是否在同一个集合中

重要思想：用集合中的一个元素代表整个集合。集合中的元素采用树结构组织。

如果查询操作很频繁的话（逼近或者超过$O(N)$），则该算法的时间复杂度可以认为是$O(1)$

详解：https://zhuanlan.zhihu.com/p/93647900

基本结构：

```python
class DSU:
    # 初始化时每个集合的代表元素（树的根节点）是自身
    def __init__(self, N):
        self.fa = [i for i in range(N)]
    # 如果发现父节点是自身，则该节点为根节点，否则继续往上找    
    def find(self, k):
        if self.fa[k] == k:
            return k
        return self.find(self.fa[k])
    # 先找到根节点，然后让一个根节点的父节点指向另一个根节点即可
    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        if x != y:
            self.fa[y] = x
        return
```

优化版：

1. 按秩合并：将深度小的树合并到深度大的树上。通过调整树型结构来提高效率。扁平化树形结构，防止出现链型结构导致树过深，增加查询的开销。

   实现方法：维护一个数组记录树的深度，合并的时候将较浅的树的根节点指向较深的树的根节点

   ```python
   class DSU:
       def __init__(self, N):
           self.fa = [i for i in range(N)]
           self.depth = [1 for i in range(N)]
           
       def find(self, k):
           if self.fa[k] == k:
               return k
           return self.find(self.fa[k])
       
       def union(self, a, b):
           x = self.find(a)
           y = self.find(b)
           xh = self.depth[x]
           yh = self.depth[y]
           if x == y:
               return
           if xh > yh:
               self.fa[y] = x
           elif xh == yh:
               self.depth[x] = max(self.depth[x], self.depth[y]+1)
               self.fa[y] = x      
           else:
               self.fa[x] = y
   ```

2. 每次查找后都更新父节点信息，将任意节点的父节点都指向其根节点

   ```python
   class DSU:
       def __init__(self, N):
           self.fa = [i for i in range(N)]
           
       def find(self, k):
           if self.fa[k] == k:
               return k
           self.fa[k] = self.find(self.fa[k])
           return self.fa[k]
       
       def union(self, a, b):
           x = self.find(a)
           y = self.find(b)
           if x != y:
               self.fa[y] = x
           return
   ```


## 最小生成树

引例：一个有 n 户人家的村庄，有 m 条路**相互**连接着。村里现在要修路，每条路都有一个成本价格，现在请你帮忙计算下，最少需要花费多少钱，就能让这 n 户人家连接起来。

本质上是**求该连通图的最小生成树**。

最小生成树是**连通加权无向图**中一棵**权值最小**的生成树，对于不连通的图，每一个连通分量都有一棵属于自己的最小生成树，合称**最小生成森林**。

求最小生成树的算法主要有Kruskal算法和Prim算法。假设图中共V个节点，E条边

### Kruskal算法

着眼于边，将边按权值升序排列，依次加入生成树中。如果生成环则略过当前边（DSU判断）。直到最小生成树中含有V-1条边为止。

因为后面循环体遍历边时时间复杂度最大为$O(E)$，所以该算法的时间复杂度为$O(ElogE)$（排序的时间复杂度）。最后循环取边的过程可以加入计数变量count，每加入一条边count+1，当count==V-1时结束，减少循环次数。

在这个算法中，最重要的一环就是判断两个节点在不在同一个集合内，很多人想，那你直接用一个set来保存不就好了？这是思路显然不行，因为假设A和其他三个节点相连，同属一个集合，而B和另外三个节点相连，同属一个集合，那么我们将A和B取并集时，是将这两组数据合并一起的，如果使用set，那么当数据量大时还需要遍历，复杂度就会很高，因此使用并查集就会效率快很多了！

```python
#coding:utf-8
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 返回最小的花费代价使得这n户人家连接起来
# @param n int整型 n户人家的村庄
# @param m int整型 m条路
# @param cost int整型二维数组 一维3个参数，表示连接1个村庄到另外1个村庄的花费的代价
# @return int整型
#
class DSU:
    def __init__(self, n):
        self.fa = [i for i in range(n)]       # 每个节点点对应的父节点
        self.depth = [1 for i in range(n)]    # 以每个节点为根节点的子树的深度
        
    def find(self, x):
        '''
        寻找x节点的根节点
        '''
        
        if x == self.fa[x]:                   # 如果当前节点的父节点是自己
            return x                          # 则该节点是根节点

        return self.find(self.fa[x])          # 否则，找父节点的根节点
                         
    def union(self, x, y):
        '''
        合并两个节点，先寻找两个节点对应的根节点，然后根据根节点对应的树的深度按秩合并
        '''
        
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:                           # 根节点相同。不需要合并
            return
        
        if self.depth[root_x] < self.depth[root_y]:    # 比较根节点对应的树的深度
            self.fa[root_x] = root_y                   # 小树向大树合并
            
        elif self.depth[root_x] == self.depth[root_y]: # 两根节点对应树高相同时
                                                       # 可任选一个节点为根节点
                                                       # 但要更新根节点树高
            self.depth[root_y] = max(
                self.depth[root_y], 
                self.depth[root_x] + 1
                )                                      # 更新根节点树高
            self.fa[root_x] = root_y
            
        else:
            self.fa[root_y] = root_x

class Solution:
    def miniSpanningTree(self , n , m , cost ):
        # write code here
        dsu = DSU(m)
        cost.sort(key=lambda x: x[2])
        cnt, res = 0, 0
        for a, b, value in cost:
            if dsu.find(a-1) == dsu.find(b-1):
                continue
            res += value
            dsu.union(a-1, b-1)
            cnt += 1
            if cnt == n - 1:
                break
        return res
```

### Prim 算法

Prim算法的每一步都会为一棵生长中的树添加一条边。该树最开始只有一个顶点，然后会添加V-1条边。每次总是选取可行边中权值最小的加入到树中，并解锁新节点。
构建图的时间复杂度为$O(m)$, 空间复杂度为$O(m)$。遍历部分借助小根堆，时间复杂度为$O(mlogm)$

```python
#coding:utf-8
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 返回最小的花费代价使得这n户人家连接起来
# @param n int整型 n户人家的村庄
# @param m int整型 m条路
# @param cost int整型二维数组 一维3个参数，表示连接1个村庄到另外1个村庄的花费的代价
# @return int整型
#

class small_root_heap(object):
    '''
    构建小根堆
    '''

    def __init__(self):
        self.arr = []
        self.heap_size = 0
        
    def heapify(self, index):
        '''
        从index位置开始向下heapify
        '''

        left = 2 * index + 1
        while left < self.heap_size:
            if left + 1 < self.heap_size and self.arr[left] > self.arr[left + 1]:
                smaller = left + 1
            else:
                smaller = left
            if self.arr[smaller] >= self.arr[index]:
                break
            else:
                self.arr[smaller], self.arr[index] = self.arr[index], self.arr[smaller]
                index = smaller
                left = 2 * index + 1
                
    def heapinsert(self, index):
        '''
        从index位置开始向上heapinsert
        '''

        p = (index - 1) >> 1
        while p >= 0 and self.arr[p] > self.arr[index]:
            self.arr[p], self.arr[index] = self.arr[index], self.arr[p]
            index = p
            p = (index - 1) >> 1
            
    def add(self, value):
        '''
        向堆中添加元素：将元素加入到数组尾部并对尾部元素heapinsert，堆大小加一
        '''

        self.arr.append(value)
        self.heapinsert(self.heap_size)
        self.heap_size += 1
        
    def pop(self):
        '''
        将数组首尾元素交换，弹出尾部元素（最值），然后从头部进行heapify，堆大小减一
        '''

        self.arr[0], self.arr[self.heap_size - 1] = self.arr[self.heap_size - 1], self.arr[0]
        small_root = self.arr.pop()
        self.heap_size -= 1
        self.heapify(0)
        return small_root
            
        
class Node(object):
    def __init__(self, value):
        self.value = value                  # value是该点的编号，唯一标识
        self.edges = set()                  # 以该点为起点的边
    
class Edge(object):
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        

    # 边要放入小根堆中进行比较，所以要定义魔法函数

    def __lt__(self, other):
        return self.weight < other.weight
    def __ge__(self, other):
        return self.weight >= other.weight
    def __gt__(self, other):
        return self.weight > other.weight
    
            
class Graph(object):
    def __init__(self, n):
        self.nodes = [Node(i + 1) for i in range(n)]     # 给所有的点编号

class Solution:
    def miniSpanningTree(self, n, m, cost):
        
        # 构建图
        graph = Graph(n)
        for a, b, weight in cost:
            a_node, b_node = graph.nodes[a - 1], graph.nodes[b - 1]
            a_node.edges.add(Edge(a_node, b_node, weight))
            b_node.edges.add(Edge(b_node, a_node, weight))
        
        # 从任意节点开始，将其临边放入小根堆
        heap = small_root_heap()
        seen = set()
        node = graph.nodes.pop()
        seen.add(node.value)
        for edge in node.edges:
            heap.add(edge)

        # 每次从小根堆中取当前权重最小的边
        # 如果该边对应的to_node没出现过
        # 将该边加入最小生成树中
        # 该边的to_node解锁的邻边放入小根堆中
        # 循环，直到加入n个顶点，即cnt==n
        res, cnt = 0, 1
        while heap.heap_size:
            edge = heap.pop()
            if edge.to_node.value not in seen:

                # 该边加入最小生成树
                seen.add(edge.to_node.value)
                res += edge.weight
                cnt += 1

                # 判断边数是否足够
                if cnt == n:
                    break
                
                # 将该边的to_node解锁的邻边加入小根堆
                for next_edge in edge.to_node.edges:
                    heap.add(next_edge)
        return res
```

## Dijkstra算法

适用于没有权值为负数的边。否则锁死操作会有问题。

规定出发点，求出发点到图中其他点的最短距离。每次从未确定节点中取一个与起点距离最短的点，将它归类为已确定节点，并用它更新从起点到其他所有未确定节点的距离。直到所有点都被归类为已确定节点。

为什么选距离最短节点作为当前节点：所有未确定节点中与起点距离最短的点，不可能被其它未确定节点更新。所以当前节点可以被归类为已确定节点。用他去更新其他未确定节点。

网络延迟问题：有 n 个网络节点，标记为 1 到 n。给你一个列表$ times$，表示信号经过 有向 边的传递时间。 $times[i] = (ui, vi, wi)$，其中$ ui $是源节点，$vi$ 是目标节点，$ wi$ 是一个信号从源节点传递到目标节点的时间。现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

```python
class Solution(object):
    def networkDelayTime(self, times, n, k):
        """
        :type times: List[List[int]]
        :type n: int
        :type k: int
        :rtype: int
        """
        def getNode():
            # 从distanceMap中取当前未locked的distance最小的节点，没有返回None
            minNode = None
            minDistance = float('inf')
            for key, value in distanceMap.items():
                if key not in locked and value < minDistance:
                    minDistance = value
                    minNode = key
            return minNode 

        def dijkstra(k):
            # 从K出发到所有点的最短距离
            minNode = getNode()
            while minNode:
                distance = distanceMap[minNode]  # 当前节点的距离
                for edge in graph[minNode - 1]:
                    to_node = edge[0]
                    if to_node not in distanceMap:  # 不在表中说明该to_node距离无穷大
                        distanceMap[to_node] = distance + edge[1]
                    else:                           # 在表中看是否可以更新该to_node的距离
                        distanceMap[to_node] = min(distanceMap[to_node], distance + edge[1])
                # 当前minNode操作结束后将其锁定，然后取新的minNode
                locked.add(minNode) 
                minNode = getNode()

        # 构建邻接表
        graph = [[] for _ in range(n)]
        for lst in times:
            graph[lst[0] - 1].append(lst[1:])
        
        # 用起始点初始化distanceMap
        distanceMap = {k: 0} # key:目标点 value：最短距离
        locked = set() # 操作过的节点锁住，以后不再操作
        dijkstra(k)
        if len(distanceMap.keys()) < n:
            return -1
        return max(distanceMap.values())
```

时间复杂度$O(m+n^2)$：$m$-构建邻接表要遍历所有边;$n^2$-遍历所有结点，每次取最小权值边要遍历distanceMap。

空间复杂度$O(n^2)$，构建邻接表。

其中，取最小权值边的函数可以进行堆优化，将解锁的边放入小根堆，每次就不需要遍历取最小权值边了，时间复杂度降到$O(mlogm)$（假设m条边）

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        
        # 构建邻接表
        graph = [[] for _ in range(n)]
        for u, v, w in times:
            graph[u - 1].append((w, v - 1))

        # 初始化距离表
        distmap = [float('inf') for _ in range(n)]
        distmap[k - 1] = 0

        # 初始化小根堆
        small_root_heap = [(0, k - 1)]

        # locked用于存储锁定的点
        locked = set()

        while small_root_heap:

            min_dist, min_node = heappop(small_root_heap)   # 弹出最短距离、最近节点
            locked.add(min_node)                            # 锁定该节点

            # 遍历当前节点解锁的新节点（邻接节点）
            for edge, to_node in graph[min_node]:

                # 如果邻接节点锁定或者距离不需要更新，跳过
                if (to_node in locked) or (edge + min_dist >= distmap[to_node]):
                    continue
                
                # 更新邻接节点距离， 并将该邻接节点加入小根堆
                distmap[to_node] = edge + min_dist
                heappush(small_root_heap, (distmap[to_node], to_node))
                
        if float('inf') in distmap:
            return -1
        return max(distmap)
```



# 前缀树和贪心算法

前缀树实现

```python
class TrieNode:
    def __init__(self):
        self.pre = 0    # 以self.root到当前字母为prefix的字符串个数
        self.end = 0    # self.root到当前字母的字符串个数
        self.nexts = [None for _ in range(26)]  # 当前节点下一个节点，如果可能性很多可以用哈希表的结构

class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        # 插入数据
        :type word: str
        :rtype: None
        """
        if not word:
            return
        node = self.root
        node.pre += 1
        for i in range(len(word)):
            index = ord(word[i]) - ord('a')
            if not node.nexts[index]:
                node.nexts[index] = TrieNode()
            node = node.nexts[index]
            node.pre += 1
        node.end += 1

    def search(self, word):
        """
        # 查询word在前缀树中的数量
        :type word: str
        :rtype: int
        """
        if not word:
            return 0
        node = self.root
        for i in range(len(word)):
            index = ord(word[i]) - ord('a')
            if not node.nexts[index]:
                return 0
            node = node.nexts[index]
        return node.end

    def startsWith(self, prefix):
        """
        # 查询前缀树中包含前缀prefix的单词的数量
        :type prefix: str
        :rtype: int
        """
        if not prefix:
            return 0
        node = self.root
        for i in range(len(prefix)):
            index = ord(prefix[i]) - ord('a')
            if not node.nexts[index]:
                return 0
            node = node.nexts[index]
        return node.pre
    
    def delete(self, word):
        '''
        # 从前缀树中删除一个word
        :type prefix: str
        ''' 
        if not self.search(word):
            return
        node = self.root
        node.pre -= 1
        for i in range(len(word)):
            index = ord(word[i]) - ord('a')
            node.nexts[index].pre -= 1
            if not node.nexts[index].pre:
                node.nexts[index] = None
                return
            else:
                node = node.nexts[index]
        node.end -= 1
  
# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

贪心算法：通过每一步最优来实现整体最优，通过每一个位置的局部最优来实现全局最优。

贪心算法不一定能实现全局算法，关键在于贪心策略的选择。

例题

1. 会议室问题：给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。（会议室问题，一天最多安排几个会议）

   ```python
       def eraseOverlapIntervals(self, intervals):
           """
           # 贪心策略：每次选最先结束的安排（区间右边界）
           :type intervals: List[List[int]]
           :rtype: int
           """
           intervals.sort(key = lambda x : x[1])
           end = intervals[0][1]
           res = 0
           for i in range(1, len(intervals)):
               if intervals[i][0] < end:
                   res += 1
                   continue
               end = intervals[i][1]
           return res
   ```

   

2. IPO问题：给你 n 个项目。对于每个项目 i ，它都有一个纯利润 profits[i] ，和启动该项目需要的最小资本 capital[i] 。

   最初，你的资本为 w 。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。

   总而言之，从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本 ，并输出最终可获得的最多资本。

   ```python
       def findMaximizedCapital(self, k, w, profits, capital):
           """
           :type k: int
           :type w: int
           :type profits: List[int]
           :type capital: List[int]
           :rtype: int
           """
           # 如果一开始所有项目都解锁，直接返回本金加上最大的k个利润即可
           if max(capital) <= w:
               return w + sum(nlargest(k, profits))
           # 将所有项目按照成本排序
           programs = [(profits[i], capital[i]) for i in range(len(profits))]
           programs.sort(key = lambda x: x[1])
           heap = []
           # 当前本金可以解锁的项目的利润加入大根堆，每次从大根堆弹出最大值更新本金
           while k:
               while programs and programs[0][1] <= w:
                   heappush(heap, -programs.pop(0)[0])
               # 大根堆为空，说明没有可做项目了
               if not heap:
                   return w
               w -= heappop(heap)
               k -= 1
           return w
   ```

3. 最大数问题：给定一组非负整数 `nums`，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数

   贪心策略：组合较大的顺序排列。如果$xy>yx$，则$x$在前$y$在后。

   ```python
       def largestNumber(self, nums):
           """
           :type nums: List[int]
           :rtype: str
           """
           # 自定义比较器：if xy > yx: x 在 y 之前
           def cmp(x, y):
               if int(str(x) + str(y)) > int(str(y) + str(x)):
                   return -1
               else:
                   return 1
           nums.sort(cmp)
           res = ''
           for num in nums:
               res += str(num)
           # 考虑首位为0的情况
           if not res.lstrip('0'):
               return '0'
           return res.lstrip('0')
   ```

4. 分金条的最小花费（哈夫曼树）：

   ```python
       def gold(self, nums):
           heap = []
           for x in nums :
               heappush(heap, x)
           res = 0
           while len(heap) > 1:
               cur = heappop(heap) + heappop(heap)
               res += cur
               heappush(heap, cur)
           return res
   ```

# 暴力递归

暴力递归就是尝试。算法概述：

1. 把问题转化为规模缩小的同类问题的子问题
2. 有明确的不需要继续进行递归的条件（base case）
3. 有当得到了子问题的结果之后的决策过程
4. 不记录每一个子问题的解

## 汉诺塔问题

我们有由底至上为从大到小放置的 n 个圆盘，和三个柱子（分别为左/中/右即left/mid/right），开始时所有圆盘都放在左边的柱子上，按照汉诺塔游戏的要求我们要把所有的圆盘都移到右边的柱子上，要求一次只能移动一个圆盘，而且大的圆盘不可以放到小的上面。

```python
    def getSolution(self, n):
        # write code here
        res = []
        def move(i, f, t, o):
            # basecase：直接打印即可
            if i == 1:
                res.append('move from ' + f + ' to ' + t)
                return
            # 先将1~i-1从from移动到other
            move(i-1, f, o, t)
            # 将i(最底下的)从from移动到to，只移动一个直接打印
            res.append('move from ' + f + ' to ' + t)
            # 将1~i-1从other移动到to
            move(i-1, o, t, f)
        move(n, 'left', 'right', 'mid')
        return res
```

## 字符串的全部子序列

```python
    def getAllSubs(self , s):
        res = []
        path = []
        def process(s, path):
            if not s:
                res.append(''.join(path))
                return
            # 不要当前值
            process(s[1:], path)
            # 要当前值
            path.append(s[0])
            process(s[1:], path)
            # 因为两个process共用一个path,所以这里必须pop,否则会污染第一个process
            path.pop()
        process(s, path)
        return res
```

## 字符串的全排列

```python
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def process(i, nums):
            # basecase:当遍历到nums列表尾，将当前nums加入到res
            if i == l:
                res.append([_ for _ in nums])
                return
            # seen用来记录在i位置出现过的数字，用来去重
            seen = set()
            # i位置之后的每一个数都有可能出现在i位置上
            for j in range(i, l):
                # 去重
                if nums[j] in seen:
                    continue
                seen.add(nums[j])
                # j遍历i位置后的每一个数，将其放在i位置上
                nums[i], nums[j] = nums[j], nums[i]
                # i后移处理下一位
                process(i+1, nums)
                # 维持nums列表不变
                nums[i], nums[j] = nums[j], nums[i]

        res = []
        l = len(nums)
        process(0, nums)
        return res
```

## 预测赢家问题

给你一个整数数组 nums 。玩家 1 和玩家 2 轮流进行自己的回合，玩家 1 先手。开始时，两个玩家的初始分值都是 0 。每一回合，玩家从数组的任意一端取一个数字（即，nums[0] 或 nums[nums.length - 1]），取到的数字将会从数组中移除（数组长度减 1 ）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。

如果玩家 1 能成为赢家，返回 true 。如果两个玩家得分相等，同样认为玩家 1 是游戏的赢家，也返回 true 。你可以假设每个玩家的玩法都会使他的分数最大化。

```python
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        def f(start, end):
            '''
            在[start, end+1]区间上先手的最大得分
            '''
            if start == end:
                return nums[end]
            return max(nums[start] + s(start+1, end), s(start, end-1) + nums[end])
        def s(start, end):
            '''
            在[start, end+1]区间上后手的最大得分
            '''
            if start == end:
                return 0
            return min(f(start+1, end), f(start, end-1))
        return f(0, len(nums)-1) >= s(0, len(nums)-1)
```



## 解码问题

1~26 对应 A~Z，将数字字符串解码成字母字符串的所有可能有多少种

```python
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        def process(i, s):
            '''
            i之前的位置已固定，求i和之后的位置有多少种可能
            '''
            # i==l,下标已越界，返回只有一种（之前确定的）
            if i == l:
                return 1
            # 假如出现'0'，则无字母对应，返回0
            if s[i] == '0':
                return 0
            # '1'对应两种可能：s[i]单独或者s[i]s[i+1]一起，注意s[i+1]不能越界
            if s[i] == '1':
                res = process(i+1, s)
                if i+1 < l:
                    res += process(i+2, s)
                return res
            # '2'与'1'类似，注意s[i+1]的取值和越界问题
            if s[i] == '2':
                res = process(i+1, s)
                if i+1 < l and s[i+1] <= '6':
                    res += process(i+2, s)
                return res
            # '3'~'9'的情况，一个数字只对应一个字母
            return process(i+1, s)
            
        l = len(s)
        return process(0, s)
```

## 背包问题

有N件物品和一个容量为V的背包。第i件物品的价值是C[i]，重量是W[i]。求解将哪些物品装入背包可使价值总和最大

```python
import sys
line = sys.stdin.readline().strip()
N, V = map(int, line.split())
weights = []
values = []
for _ in range(N):
    line = sys.stdin.readline().strip()
    value, weight = map(int, line.split())
    weights.append(weight)
    values.append(value)

def process(i, gotweight):
    if i == N:
        return 0
    if gotweight + weights[i] > V:
        return process(i+1, gotweight)
    return max(  process(i+1, gotweight), values[i] + process(i+1, gotweight + weights[i])  )
res = process(0, 0)
print(res)
```

## N皇后问题

将 `n` 个皇后放置在$n*n$的棋盘上，并且使皇后彼此之间不能相互攻击。给你一个整数 `n` ，返回所有不同的 n 皇后问题的解决方案。

```python
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def generateBoard():
            board = []
            for i in range(n):
                line[queens[i]] = 'Q'
                board.append(''.join(line))
                line[queens[i]] = '.'
            res.append(board)
            
        def process(row, cols, lefts, rights):
            # 越界则说明当前方案可行，按照queens列表生成board
            if row == n:
                generateBoard()
                return
            # 在当前row遍历每一列col，看能否放'Q'
            for col in range(n):
                if col not in cols and (row + col) not in lefts and (row - col) not in rights:
                    queens[row] = col
                    cols.add(col)
                    lefts.add(row + col)
                    rights.add(row - col)
                    process(row + 1, cols, lefts, rights)
                    cols.remove(col)
                    lefts.remove(row + col)
                    rights.remove(row - col)
                    queens[row] = None
        
        # 记录每一行的queen所在的列
        queens = [None] * n 
        # 记录不能选取的点
        cols = set()    # 列
        lefts = set()   # 左对角线
        rights = set()  # 右对角线
        # 用于生成generateBoard的每一行
        line = ['.'] * n 

        res = []
        process(0, cols, lefts, rights)
        return res
```

位运算优化：用三个数代替cols, lefts 和 rights。利用其二进制字符串中'1'的位置标记不能放置皇后的点。提高效率，降低存储cols, lefts,rights的空间复杂度。

知识基础：

- $x\&(-x)$：可以获取x的二进制字符串中最后一个'1'
- $x\&(x-1)$：可以将x的二进制字符串中的最后一个'1'置'0'
- $bin(x-1).count('1')$：可以获取x的二进制字符串中'1'的位置（x的二进制字符串中只有一个'1'）

```python
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def generateBoard():
            board = []
            for i in range(n):
                line[queens[i]] = 'Q'
                board.append(''.join(line))
                line[queens[i]] = '.'
            res.append(board)
        
        def process(row, cols, lefts, rights):
            if row == n:
                generateBoard()
                return
            # 得到当前行所有可能放皇后的位置
            availablePosition = ((1 << n) - 1) & (~(cols | lefts | rights))
            while availablePosition:
                # 取出所有可能位置中最低位的'1'存到position中
                position = availablePosition & (- availablePosition)
                # 将最低位的'1'置'0'
                availablePosition = availablePosition & (availablePosition - 1)
                # 得到最低位'1'的下标作为col
                col = bin(position - 1).count('1')
                queens[row] = col
                # 将三个数与position（该次循环的可行位置）或运算。注意lefts要往低位移动，rights要往高位移动
                process(row + 1, cols | position, (lefts | position) >> 1, (rights | position) << 1)

        line = ['.'] * n
        queens = [None] * n
        res = []
        process(0, 0, 0, 0)
        return res
```

# 哈希函数和哈希表

## 哈希函数

1. 输入域无穷，输出阈有限。MD5：$0\sim2^{64}-1$；SHa1：$0\sim2^{128}-1$
2. same in，same out：不随机
2. diff in， same out：哈希碰撞（概率极小）
2. 任意一组输入对应的输出在输出域中的分布是均匀离散的（哈希函数最重要的性质）（离散性与均匀性本质是一回事）。而且将该组均匀分布的输出对m取模，得到的新的输出一定在$0 \sim m-1$范围上，且也是均匀分布。

哈希函数应用举例：40亿个数寻找众数，要求内存空间不超过1G。

用哈希表的话每种数至少8byte，假如40亿个数各不相同，则需要320亿byte，大约是32G，不可行。

解决思路：将40亿个数通过哈希函数映射到输出域，然后对100取模。由于哈希函数的均匀性，40亿个数的映射会均匀分布在$0\sim 99$中。在硬盘上开100个文件，根据映射结果将40亿数字写入相应的文件里，相同的数字肯定会被分到同一组里。然后分别用哈希表找每一组的众数，所占用的空间大约是$32G/100=0.32G$。

## 哈希表的实现

哈希表的经典版是通过哈希函数实现的。根据输入x通过哈希函数映射再对m取模得到的值确定x在长度为m的数组中的位置，类似于分组操作。每一组中的数据是用单向链表的形式存储的。当单向链表太长时会影响哈希表性能所以要进行扩容，即增大m的值将数据重新组织。

操作复杂度分析：

- 哈希化的复杂度为$O(1)$，如果将每个分组下的链表的长度限制的很小，则增删改查时间复杂度都近似$O(1)$。
- 扩容的时间复杂度：假如一共N个数，则扩容的次数是$O(logN)$级别（假设链长是K，则扩容次数为$O(log\frac{N}{K})$）。每一次扩容的数据量是N，所以扩容的总时间复杂度为$O(Nlog\frac{N}{K})$，分摊到每一次的时间复杂度为$O(log\frac{N}{K})$。假设链长不太小，则可以认为扩容的时间复杂度逼近$O(1)$
- 在一些语言中还有离线扩容的技术，当用户不在线时进行扩容。

哈希表应用举例：设计RandomPool结构

结构的三个功能：

- insert(key)：将key加入，要求不重复
- delete(key)：移除
- getRandom()：等概率随机返回结构中的任何一个key

思路：两个哈希表加一个整型变量实现。一个存储$key:index$，另一个存储$index:key$。整型size代表哈希表中元素个数。难点在于移除操作。移除时，用最后位置的元素补位，保证index取值范围是$0\sim size-1$

## 布隆过滤器

它实际上是一个m长度的二进制向量（bit 数组）和一系列随机映射函数（哈希函数+取模）。布隆过滤器可以用于检索一个元素是否在一个集合中。它的优点是空间效率和查询时间都远远超过一般的算法，缺点是有一定的误识别率和删除困难。

存储时对于每一个元素，分别按K个哈希函数映射到$0\sim m-1$之间的K个位置进行标记。查询时按照查询元素的K个映射结果，只要有一个结果没有标记则说明该元素不在集合中。

由于哈希碰撞，这种机制会导致有一定的误杀率。增大m会缓解这个问题，适当增大K值也会缓解这个问题，但是K过大误杀率会增大。

设计布隆过滤器：

1. 确定问题：是不是集合查询问题，是否允许一定失误率，是否要求删除操作。（判断是不是布隆过滤器问题）

2. 确定失误率p和样本量n，不需要确定单样本大小

3. 确定布隆过滤器参数理论值（向上取整）：
   $$
   m = -\frac{n*lnp}{(ln2)^2}\\
   K = ln2*\frac{m}{n}\\
   $$

4. 计算真实失误率：
   $$
   P_{true}=(1-e^{-\frac{n*K_{true}}{m_{true}}})^{K_{true}}
   $$
   

## 一致性哈希

传统哈希取模算法的痛点在于容量改变（增删节点）的时候要将当前数据全部重新组织，复杂度较高。一致性哈希抛弃了传统的取模算法，只进行哈希。一致性哈希算法使得在容量改变时数据存储的改变最小。

一致性哈希是通过一致性哈希环结构来实现的，将节点哈希化后映射到环上，按顺时针方向，节点之前的数据存储在该节点上。增删节点时，只在环上局部操作即可。

但是局部操作会带来负载不均衡的问题。可以通过虚拟节点解决这个问题。虚拟节点是指每一个物理服务器会映射好多虚拟服务器在环上，这样环上每一部分各个服务器的比例基本是均衡的。增删节点的时候，该节点对应的物理服务器的虚拟节点也是均匀分布在环上的。具体点说：均匀的从环上抢数据或者仍数据，这样就可以保证负载的均衡性。

# KMP字符串匹配

字符串匹配问题：给你两个字符串 str1 和 str2 ，请你在 str1 字符串中找出 str2 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

暴力遍历方法：假设str1长度为n，str2长度为m，则时间复杂度为$O(n*m)$

问题：暴力遍历的问题在于，当发现失配字符时，str1和str2都回到起始位置重新开始比较，没有利用之前匹配的信息。

优化：KMP算法中，发现失配字符时，str1不需要回退，str2直接回退到指定位置（该位置由next数组确定）。时间复杂度为$O(m+n)$

详解：https://www.zhihu.com/question/21923021

代码实现：

```python
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        def getNext():
            if l2 == 1:
                return [-1]
            nexts = [0 for _ in range(l2)]
            nexts[0] = -1          
            i = 2
            index = 0   # 代表前缀的后一个位置，其值等于前缀的长度
            while i < l2:
                if needle[index] == needle[i-1]:
                    # 如果前缀后一个位置的字符与i-1位置的字符匹配，next[i]直接等于index+1
                    # 然后将index和i后移
                    nexts[i] = index + 1
                    i += 1
                    index += 1
                elif index > 0:
                    # 如果不匹配但是index>0，更新index
                    index = nexts[index]
                else:
                    # 如果index==0，将nexts[i]置0，i后移
                    nexts[i] = 0
                    i += 1
            return nexts
        
        l1, l2 = len(haystack), len(needle)
        if not l2:
            return 0
        if l1 < l2:
            return -1
        nexts = getNext()
        i1, i2 = 0, 0
        while i1 < l1 and i2 < l2:
            if haystack[i1] == needle[i2]:
                # 匹配，都后移
                i1 += 1
                i2 += 1
            elif i2 == 0:
                # 不匹配，i2已经跳回到开始位置，i1后移重新开始匹配
                i1 += 1
            else:
                # 不匹配，但是i2没有回到开始位置，往前跳
                i2 = nexts[i2]
        if i2 == l2:
            return i1 - l2
        return -1
```

next数组是KMP的精髓

# Manacher算法

引例：求字符串中最长回文子串的长度，要求时间复杂度$O(N)$

经典处理方法：在原字符串中，任意两个字符串之间以及字符串的首尾插入任意特殊字符（比如#），然后对新字符串遍历，以$i$位置的字符为轴向左右检验，时间复杂度为$O(N^2)$

优化方法：

对原字符串处理，对处理后的字符串操作。准备一个回文半径数组，存储每个位置的回文半径。两个变量分别存储回文最右边界的位置R及其对应的轴C。回文半径数组是精髓。

对于第$i$个位置，有以下几种情况：

1. $i$在右边界外部：从$i$开始暴力外扩，更新R和C

2. $i$在右边界内部，假设$i$关于C的对称点为$i^{’}$：

   2.1. 以$i^{’}$为轴的回文区域在以C为轴的回文区域内部：$i$的回文区域与$i^{’}$关于C对称

   2.2. 以$i^{’}$为轴的回文区域左边界在以C为轴的回文区域左边界之外：$i$的回文区域半径为$i$到右边界R

   2.3. 以$i^{’}$为轴的回文区域左边界与以C为轴的回文区域左边界重合：$i$的回文区域半径至少为$i$到右边界R，从R处开始向外扩以确定回文半径，更新R和C

代码：

```python
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def getStr():
            '''
            处理字符串
            '''
            res = '#'
            for i in range(len(s)):
                res += s[i]
                res += '#'
            return res

        s_new = getStr()
        l = len(s_new)        
        parr = [0 for _ in range(l)] # 记录回文半径
        C, R = -1, -1                # 记录最右回文边界对应的轴位置和边界位置
        res, index = -1, -1          # 记录最大回文半径及其对应的轴位置
        for i in range(l):
            # 初始化当前位置的回文半径
            if i < R:                # 当前位置在最右边界内，对应2
                parr[i] = min(parr[2*C - i], R - i)
            else:                    # 当前位置在最右边界外，对应1
                parr[i] = 1
            
            # 对情况2进行细分
            while i + parr[i] < l and i - parr[i] >= 0:       # 检查是否越界
                if s_new[i + parr[i]] == s_new[i - parr[i]]:  # 对应2.3
                    parr[i] += 1 
                else:                                         # 对应2.1和2.2
                    break
            # 更新C和R
            if i + parr[i] > R: 
                R = i + parr[i]
                C = i
            if res < parr[i]:
                res = parr[i]
                index = i
        return s_new[index - res + 1: index + res].replace('#', '')
```

# 滑动窗口最值

滑动窗口：左边界和右边界都只能往右移动，左边界不能超过右边界。

引例：确定滑动窗口的最大值

给你一个整数数组$nums$，有一个大小为$k$的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的$k$个数字。滑动窗口每次只向右移动一位。返回滑动窗口中的最大值

代码：

```python
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        def moveR():
            '''
            移动窗口右边界，维持queue中的严格倒序（小于等于当前值的下标弹出扔掉）
            '''
            while queue and nums[self.p2] >= nums[queue[-1]]:
                queue.pop()
            queue.append(self.p2)
            self.p2 += 1
        def moveL():
            '''
            移动窗口左边界，queue[0]是当前值的下标直接弹出，否则说明当前值的下标之前移动右边界时已经扔掉
            '''
            if self.p1 == queue[0]:
                queue.pop(0)
            self.p1 += 1
        
        queue = []
        res = []
        self.p1, self.p2 = 0, 0
        while self.p2 < k:
            moveR()
        res.append(nums[queue[0]])
        while self.p2 < len(nums):
            moveR()
            moveL()
            res.append(nums[queue[0]])
        return res
```

# 单调栈

求数组arr中当前值两侧最近的较大值。

准备一个栈存放下标，维持栈中的下标对应的数据为栈底向栈顶递增。

遍历给定数组arr。对于当前数据arr[i]，如果小于栈顶的数据直接压入栈即可。否则，将栈顶小于当前数据的数依次弹出，**弹出过程中**，弹出的数据左边最近的较大值即为弹出后栈顶的元素，右边最近的较大值是当前数据。最后将当前数据压入栈顶。对于有重复值的情况，将重复值的下标压在同一个数组里存入栈中即可即可，比如$[[0],[1,2],[3]]$

应用：数组中累计和与最小值的乘积称为指标A。给定一个正数数组，请返回子数组中，指标A最大的值。

思路：用一个递减单调栈解决。遍历数组，以当前数作为最小值，用单调栈求出其左右最近的较小值作为左右边界，将边界中的数组求和并乘上当前值即为当前数组的A。

代码：

```python
    def maxSumMinProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        left = [0] * l        # <= 当前值的左边界
        right = [l - 1] * l   # >当前值的右边界
        stack = []
        
        # 遍历数组，利用递增单调栈确定任意数为最小值的子数组的左右边界
        for i in range(l):
            while stack and nums[stack[-1]] >= nums[i]:
                right[stack[-1]] = i - 1   # 维持栈底到栈顶递增，弹出之前确定右边界
                stack.pop()                # 一直没弹出的由于递增性质，其右边界默认为l-1
            if stack:                      # 入栈之前确定左边界为栈顶元素+1，栈空默认是0
                left[i] = stack[-1] + 1
            stack.append(i)                #将当前下标入栈，入栈的元素左边界都已经确定
        
        # 求前缀和
        pre = [0]
        for num in nums:
            pre.append(pre[-1] + num)
        res = 0
        
        # 利用前缀和求快速求最大值
        for i in range(l):
            temp = (pre[right[i] + 1] - pre[left[i]]) * nums[i]
            res = max(res, temp)
        return res % (10 ** 9 + 7)
```

# 动态规划

## 引例

假设有排成一行的N个位置，记为1~N，开始时机器人在M位置，机器人可以往左或者往右走，如果机器人在1位置，那么下一步机器人只能走到2位置，如果机器人在N位置，那么下一步机器人只能走到N-1位置。规定机器人只能走k步，最终能来到P位置的方法有多少种。由于方案数可能比较大，所以答案需要对1e9+7取模。

暴力递归方法：

```python
import sys
line = sys.stdin.readline().strip()
N, M, K, P = map(int, line.split())
MOD = 10**9 + 7
def process(cur, rest):
    if rest == 0:
        if cur == P:
            return 1
        return 0
    if cur == 1:
        return process(cur + 1, rest - 1)
    if cur == N:
        return process(cur - 1, rest - 1)
    return process(cur + 1, rest - 1) + process(cur - 1, rest - 1)
print(process(M, K)% MOD)
```

暴力递归的问题很明显，对于每一步都需要调用左右位置进行递归，是一棵高度为K的二叉树，时间复杂度为$O(2^K)$。实际上，这里存在很多重复计算。对于固定的参数cur和rest，process的返回值是固定的。因此可以用一张二维表来储存起来，即记忆化搜索法：

```python
import sys
line = sys.stdin.readline().strip()
N, M, K, P = map(int, line.split())
MOD = 10**9 + 7
dp = [[None] * (N + 1)] * (K + 1)
def process(cur, rest):
    if dp[rest][cur]:    # 之前算过直接返回即可
        return dp[rest][cur]
    if rest == 0:
        if cur == P:
            dp[rest][cur] = 1
        else:
            dp[rest][cur] = 0
        return dp[rest][cur]
    if cur == 1:
        dp[rest][cur] = process(cur + 1, rest - 1)
    elif cur == N:
        dp[rest][cur] = process(cur - 1, rest - 1)
    else:
        dp[rest][cur] = process(cur + 1, rest - 1) + process(cur - 1, rest - 1)
    return dp[rest][cur]
print(process(M, K)% MOD)
```

记忆化搜索法保证二维表中的值只被计算过一次，时间复杂度为$O(K*N)$。

记忆化搜索法中没有考虑位置依赖，还是利用了递归的思想去确定二维表中的值。如果用位置依赖代替递归确定二维表中的值就成了动态规划：

```python
import sys
line = sys.stdin.readline().strip()
N, M, K, P = map(int, line.split())
MOD = 10**9 + 7
dp = [[0] * (N + 1) for i in range(K + 1)]
dp[0][P] = 1
#print(dp)
for i in range(1, K + 1):
    for j in range(1, N + 1):
        if j == 1:
            dp[i][j] = dp[i-1][j+1]
        elif j == N:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = (dp[i-1][j-1] + dp[i-1][j+1])
        if i == K and j == M:
            break
print(dp[K][M] % MOD)
```

具体的记忆化方法改严格动态规划的流程：

1. 分析递归中可变参数的数目，确定记忆表维度
2. 标出要计算的终止位置
3. 根据递归的basecase初始化记忆表
4. 推导位置依赖并确定表的填充方向（根据递归调用规则）

## 预测赢家问题

递归写法见上文，这里根据递归写法给出动态规划写法

```python
class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        l = len(nums)
        # 构造先手后手矩阵并初始化
        f = [[0] * l for _ in range(l)]
        s = [[0] * l for _ in range(l)]
        for i in range(l):
            f[i][i] = nums[i]
        
        # 平行于对角线进行填充
        for index in range(1, l):
            i, j = 0, index
            while j < l:
                f[i][j] = max(nums[i] + s[i+1][j], s[i][j-1] + nums[j])
                s[i][j] = min(f[i+1][j], f[i][j-1])
                i += 1
                j += 1
        # 最后需要的结果是整个序列上的，即i,j分别指向头尾
        return f[0][l-1] >= s[0][l-1]
```

递归的方法每一个数字都有两种情况，即被两个不同的玩家拿到，所以时间复杂度为$O(2^n)$。动态规划的方法只需要将两个上三角矩阵的值填充即可，时间复杂度为$O(n^2)$。

## 下象棋问题

链接：https://ac.nowcoder.com/acm/problem/21560?&headNav=www
来源：牛客网

现在给定一个棋盘，大小是n*m,把棋盘放在第一象限，棋盘的左下角是(0,0),右上角是(n - 1, m - 1);
 小乐乐想知道，一个马从左下角(0, 0)开始，走了k步之后，刚好走到右上角(n - 1, m - 1)的方案数。

暴力递归方法：

```python
import sys
line = sys.stdin.readline().strip()
n, m, k = map(int, line.split())
def process(i, j, step):
    if i < 0 or i > n or j < 0 or j > m:
        return 0
    if not step:
        if (i, j) == (n-1, m-1):
            return 1
        return 0
    return process(i+1, j-2, step-1) + process(i-1, j-2, step-1) + process(i+2, j-1, step-1)\
+ process(i+2, j+1, step-1) + process(i+1, j+2, step-1) + process(i-1, j+2, step-1)\
+ process(i-2, j+1, step-1) + process(i-2, j-1, step-1)

print(process(0, 0, k))
```

暴力递归每一步都可以有八个位置选择，时间复杂度为$O(8^k)$

动态规划方法：

```python
import sys
def getValue(i, j, step):
    if i<0 or i>=n or j<0 or j>=m:
        return 0
    return dp[step][i][j]
while True:
    line = sys.stdin.readline().strip()
    if not line:
        break
    n, m, k = map(int, line.split())
    dp = [[[0] * m for _ in range(n)] for _ in range(k+1)]
    dp[0][0][0] = 1
    for step in range(1, k+1):
        for i in range(n):
            for j in range(m):
                dp[step][i][j] += getValue(i-2, j-1, step-1)
                dp[step][i][j] += getValue(i-2, j+1, step-1)
                dp[step][i][j] += getValue(i-1, j-2, step-1)
                dp[step][i][j] += getValue(i-1, j+2, step-1)
                dp[step][i][j] += getValue(i+2, j-1, step-1)
                dp[step][i][j] += getValue(i+2, j+1, step-1)
                dp[step][i][j] += getValue(i+1, j-2, step-1)
                dp[step][i][j] += getValue(i+1, j+2, step-1)
    print(dp[k][n-1][m-1])
```

动态规划方法填充一个三维数组即可，时间复杂度为$O(n*m*k)$

## 零钱兑换

给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。

暴力递归：

```python
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        def process(index, rest):
            '''
            index:当前所处的coins数组中的位置
            rest:当前所需要凑出的数值
            return：coins[index:]可以凑出rest的方法数量
            '''
            if index == l:
                # 遍历完数组，如果恰好剩余金额rest为零则方法成立，返回1，否则返回0
                return rest == 0
            nums, ways = 0, 0 # 当前面额的数量，当前方法的数量
            while rest - nums * coins[index] >= 0:
                ways += process(index + 1, rest - nums * coins[index])
                nums += 1
            return ways
        l = len(coins)
        return process(0, amount)
```

动态规划：

```python
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        l = len(coins)
        dp = [[0] * (amount + 1) for _ in range(l+1)]  # (l+1)*(amount+1):横坐标index，纵坐标rest
        dp[l][0] = 1                                   # basecase:index=l时只有rest=0时取1，方案成立
        
        # 根据递归规则确定填充方向和依赖关系
        for i in range(l-1, -1, -1):
            for j in range(amount+1):
                num = 0
                while j - num * coins[i] >= 0:
                    dp[i][j] += dp[i+1][j-num*coins[i]] #将递归行为替换成dp拿值即可
                    num += 1
        return dp[0][amount]                          # 最后返回index=0, rest=amount的方案数
```

dp的维度为$(l*amount)$,每次确定dp数组中的一个值需要进行枚举，枚举的复杂度为$O(amount)$。所以整体的时间复杂度为$O(l*amount^2)$

动态规划优化版：在确定dp数组中的值时可以不进行枚举。假如当前位置为(i, rest)，当前位置的值由$(i+1,rest),(i+1,rest-coins[i]),(i+1,rest-2*coins[i])...$确定。考虑位置（i，rest-coins[i]），它的值由$(i+1,rest-coins[i]),(i+1,rest-2*coins[i])...$确定，二者只差一个$(i+1,rest)$。因此可以得到：
$$
dp[i][rest]=dp[i][rest-coins[i]]+dp[i+1][rest]
$$
根据上式可直接得到当前位置的值，省去了枚举过程。代码如下：

```python
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        l = len(coins)
        dp = [[0] * (amount + 1) for _ in range(l+1)]  # (l+1)*(amount+1):横坐标index，纵坐标rest
        dp[l][0] = 1                                   # basecase:index=l时只有rest=0时取1，方案成立
        
        # 根据递归规则确定填充方向和依赖关系
        for i in range(l-1, -1, -1):
            for j in range(amount+1):
                dp[i][j] = dp[i+1][j]                  # 先初始化当前值
                if j - coins[i] >= 0:                  # 判断一下是否越界
                    dp[i][j] += dp[i][j-coins[i]]
        return dp[0][amount]
```

枚举行为直接用临近位置计算，这样时间复杂度就变成了$O(l*amount)$。

## 编辑距离
求解字符串s1和s2之间的最小编辑距离，并回溯得到最优编辑操作
```python
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

```






## 有效括号
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号 子串 的长度。

注意：有效的定义是**格式正确且连续**。如“（）（（）”最长为3，因为中间不连续。

解法一：动态规划
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # 动态规划
        if not s:
            return 0
        
        dp = [0 for _ in s] # 以dp[i] 代表 以s[i]的最长连续有效括号的长度
        if s[:2] == "()":
            dp[1] = 2
        
        for i in range(2, len(s)):
            
            if s[i] == "(":
                dp[i] = 0 # 以“（”结尾不满足“格式正确”
            
            else:
                # 与前一个“（”匹配，相当于多一组括号“（）”，直接加2
                if s[i - 1] == "(":
                    dp[i] = dp[i - 2] + 2
                
                # 否则，需要判断前面子串的前一个是否是“（”。前面子串的长度为dp[i - 1]
                # 因此前面子串的前一个的下标为i - dp[i - 1] - 1
                # "()(())", "())())"
                else:
                    
                    # 先判断是否越界
                    if i - dp[i - 1] - 1 < 0:
                        dp[i] = 0
                    
                    # 判断是否匹配，如果匹配，说明当前连续括号的长度为dp[i - 1] + 2
                    # 注意还要加上前面的连续括号长度 dp[i - dp[i - 1] - 2]
                    # 例如 "()(（)）"，i=5时，dp[i - 1] = 2, dp[i - dp[i - 1] - 2] = 2
                    elif s[i - dp[i - 1] - 1] == "(": 
                        dp[i] = 2 + dp[i - 1] + dp[i - dp[i - 1] - 2]
                    
                    # 不匹配清零
                    else:
                        dp[i] = 0
        return max(dp)
```
时间复杂度和空间复杂度都是$O(n)$

方法二：栈

核心思想：“每一段子串都是以‘）’结尾的”，将前一段字串末尾的“）”的下标作为栈底，不断更新最大长度。
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        
        max_len = 0
        stack = [-1] # 最左边补一个“）”
        for i in range(len(s)):

            # 遇到“（”就往栈里加一个下标
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()

                # 如果pop出去的是“（”，说明匹配成一对括号，此时需要计算一下长度。现在栈底肯定还有一个“）”，栈不为空。
                if stack:
                    max_len = max(max_len, i - stack[-1])
                
                # 如果pop出去的是栈底的“）”，说明匹配括号失败，当前子段结束，将栈底更新成当前“）”的下标
                else:
                    stack.append(i)

        return max_len
```
时间复杂度和空间复杂度都是$O(n)$

方法三：左右计数器

思路：顺序遍历字符串，左右括号计数，相等则更新最大长度，右括号大则归零。倒序再来一遍，解决“（（）”。
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:

        max_len = 0
        left, right = 0, 0
        
        for char in s:
            if char == "(":
                left += 1
            else:
                right += 1
            if left == right:
                max_len = max(max_len, left)
            if left < right:
                left, right = 0, 0
        
        left, right = 0, 0
        for char in s[::-1]:
            if char == ")":
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left)
            if left > right:
                left, right = 0, 0
        return max_len * 2
```
时间复杂度为$O(n)$，空间复杂度为$O(1)$。

# 有序表

与哈希表的区别：有序表key是有序组织的。

有序表所有操作时间复杂度都是$O(logN)$。可以实现有序表的结构：红黑树、AVL、SB（size balance tree）、跳表（skiplist）。其中前三种的实现都是以搜索二叉树为基础的，同时考虑了平衡性（左右子树高度差限制）。

## 搜索二叉树实现
```
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
        
        # 子节点数量小于2
        if cur.left is None or cur.right is None:
            child = cur.left or cur.right
            if cur != self._root:
                if pre.left == cur:
                    pre.left = child
                else:
                    pre.right = child
        
            else:
                self._root = child
        
        # 子节点数等于2
        else:
            tmp = TreeNode(cur.right)
            while tmp.left is not None:
                tmp = tmp.left
            
            # 递归删除节点tmp
            self.remove(tmp.val)
            cur.val = tmp.val

```

搜索二叉树+搜索二叉树的左旋和右旋——>带有自平衡操作的搜索二叉树——>AVL、红黑树、SB树

搜索二叉树的左旋：头节点右子节点作为新的头节点

搜索二叉树的右旋：头节点左子节点作为新的头节点

左旋右旋操作可以提高树的平衡性。

## AVL树实现
```

```

跳表：对于每一个数据随机生成一定层数的索引（层高的概率为0.5的层高次方，即层级越高的索引产生的概率越小），跳表查询时从最高层开始，利用高层索引可以快速跳过大量数据（最底层索引存储数据），大大提高查询速度。跳表结构有点类似于二叉树结构，只不过用随机产生索引的概率来代替二叉树的分叉结构。跳表各操作的时间复杂度为$O(logN)$

# AC自动机
1.  以 Trie 的结构为基础，结合 KMP 的思想 建立的自动机，用于解决多模式匹配等任务。
2.  基础的 Trie 结构：将所有的模式串构成一棵 Trie。
3.  KMP 的思想：对 Trie 树上所有的结点构造失配指针，失配指针指向当前状态的最长后缀状态。
## 字符流问题
设计一个算法：接收一个字符流，并检查这些字符的后缀是否是字符串数组 words 中的一个字符串。
```
class TrieNode:
    def __init__(self):

        self.children = [None] * 26 # 这里更通用的做法是构建一个字典
        self.isEnd = False
        self.fail = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, seq: str):
        cur = self.root
        for char in seq:
            idx = ord(char) - ord("a")
            if not cur.children[idx]: # 这里一定要判断，之前没有才能新建，不然将之前的覆盖掉，树结构就破坏了
                cur.children[idx] = TrieNode()
            cur = cur.children[idx]
        cur.isEnd = True


class StreamChecker:

    def __init__(self, words: List[str]):

        # 初始化Trie树
        self.Trie = Trie()
        for word in set(words):
            self.Trie.insert(word)
        
        # 确定fail指针
        self.Trie.root.fail = self.Trie.root
        q = deque()
        for i in range(26):
            if self.Trie.root.children[i]:
                self.Trie.root.children[i].fail = self.Trie.root
                q.append(self.Trie.root.children[i])
            else:
                self.Trie.root.children[i] = self.Trie.root # 父节点下沉，路径压缩
        
        while q:
            node = q.popleft()

            # 当自己或者自己的某一个后缀能匹配字符串时为true，query路径压缩
            node.isEnd = node.isEnd or node.fail.isEnd
            
            for i in range(26):
                if node.children[i]:
                    node.children[i].fail = node.fail.children[i]
                    q.append(node.children[i])
                else:
                    node.children[i] = node.fail.children[i] # 路径压缩
        
        # 初始化当前节点
        self.tmp = self.Trie.root
        

    def query(self, letter: str) -> bool:
        self.tmp = self.tmp.children[ord(letter) - ord("a")]
        return self.tmp.isEnd
        
# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)
```
时间复杂度：$O(L+q)$-$L$是构建Trie树的时间复杂度，$q$是查询的时间复杂度；
空间复杂度：$O(L)$，存储Trie树。
但是上面的分析没有考虑词表的大小，英文只有26个字母但中文词表太大，所以复杂度更高，且这种方法的适用范围比较小，更普适的方法是将next换成字典结构，如下：
```
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
            return self.tmp.isEnd
        
        parent = self.tmp
        while parent.fail and not parent.fail.next.get(letter, None):
            parent = parent.fail
        self.tmp = parent.fail.next[letter] if parent.fail else self.trie.root
        if self.tmp.isEnd:
            print(self.tmp.word)
        return self.tmp.isEnd

```
这种方法适用性更广，不需要遍历整个可选字符空间，但是在fail指针确定时开销可能更大（宽度小深度大），适用于关键词等匹配