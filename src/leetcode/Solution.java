package test;


import com.sun.scenario.effect.impl.state.AccessHelper;

import java.util.*;
import java.util.stream.Collectors;

class Solution {

    /**
     * 最长公共前缀
     *
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }
        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j)) {
                    break;
                }
            }
            ans = strs[i].substring(0, j);
            if ("".equals(ans)) {
                return ans;
            }
        }
        return ans;
    }


    /**
     * 如果出现下述两种情况，交易 可能无效：
     * <p>
     * 交易金额超过 ¥1000
     * 或者，它和另一个城市中同名的另一笔交易相隔不超过 60 分钟（包含 60 分钟整）
     * 每个交易字符串transactions[i]由一些用逗号分隔的值组成，这些值分别表示交易的名称，时间（以分钟计），金额以及城市。
     * <p>
     * 给你一份交易清单transactions，返回可能无效的交易列表。你可以按任何顺序返回答案。
     * 示例 1：
     * <p>
     * 输入：transactions = ["alice,20,800,mtv","alice,50,100,beijing"]
     * 输出：["alice,20,800,mtv","alice,50,100,beijing"]
     * 解释：第一笔交易是无效的，因为第二笔交易和它间隔不超过 60 分钟、名称相同且发生在不同的城市。同样，第二笔交易也是无效的。
     * 示例 2：
     * <p>
     * 输入：transactions = ["alice,20,800,mtv","alice,50,1200,mtv"]
     * 输出：["alice,50,1200,mtv"]
     * 示例 3：
     * <p>
     * 输入：transactions = ["alice,20,800,mtv","bob,50,1200,mtv"]
     * 输出：["bob,50,1200,mtv"]
     *
     * @param transactions
     * @return
     */
    public static List<String> invalidTransactions(String[] transactions) {
        HashMap<String, List<String>> hashMap = new HashMap<>();
        List<String> result = new ArrayList<>();
        for (int i = 0; i < transactions.length; i++) {
            String transactName = transactions[i].split(",")[0];
            List<String> transactionList;
            if (hashMap.get(transactName) == null) {
                transactionList = new ArrayList<>();
            } else {
                transactionList = hashMap.get(transactName);
            }
            transactionList.add(transactions[i]);
            hashMap.put(transactName, transactionList);
        }
        for (Map.Entry<String, List<String>> transaction : hashMap.entrySet()) {
            result.addAll(compareTransaction(transaction.getValue()));
        }
        return result;
    }

    private static List<String> compareTransaction(List<String> transactions) {
        List<String> set = new ArrayList<>();
        for (int i = 0; i < transactions.size(); i++) {
            if (Integer.valueOf(transactions.get(i).split(",")[2]) > 1000) {
                set.add(transactions.get(i));
                continue;
            }
            for (int j = 0; j < transactions.size(); j++) {
                int curTime = Integer.valueOf(transactions.get(i).split(",")[1]);
                int compareTime = Integer.valueOf(transactions.get(j).split(",")[1]);
                String curCity = String.valueOf(transactions.get(i).split(",")[3]);
                String compareCity = String.valueOf(transactions.get(j).split(",")[3]);
                if (Math.abs(curTime - compareTime) <= 60 && !curCity.equals(compareCity)) {
                    set.add(transactions.get(i));
                    break;
                }
            }
        }
        return set;
    }


    /**
     * 求最长无重复子串
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> hashMap = new HashMap<>();
        int max = 0, left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (hashMap.containsKey(s.charAt(i))) {
                left = Math.max(left, hashMap.get(s.charAt(i)) + 1);
            }
            hashMap.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }


    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    /**
     * 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }


    /**
     * 分治法合并n个升序链表
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        int left = 0, right = lists.length - 1;
        return merge(lists, left, right);
    }

    private ListNode merge(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        if (left > right) {
            return null;
        }
        int middle = (left + right) / 2;
        return mergeTwoLists(merge(lists, left, middle), merge(lists, middle + 1, right));
    }


    class ListNodeComparator implements Comparable<ListNodeComparator> {

        int val;
        ListNode node;

        ListNodeComparator(int val, ListNode node) {
            this.val = val;
            this.node = node;
        }


        @Override
        public int compareTo(ListNodeComparator o) {
            return this.val - o.val;
        }
    }

    PriorityQueue<ListNodeComparator> queue = new PriorityQueue<>();

    /**
     * 使用优先级队列实现多个链表合并
     *
     * @param lists
     * @return
     */
    public ListNode mergeKListsWithPriorityQueue(ListNode[] lists) {
        for (ListNode listNode : lists) {
            if (listNode != null) {
                queue.offer(new ListNodeComparator(listNode.val, listNode));
            }
        }
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (!queue.isEmpty()) {
            ListNodeComparator curNode = queue.poll();
            tail.next = curNode.node;
            tail = tail.next;
            if (curNode.node.next != null) {
                queue.offer(new ListNodeComparator(curNode.node.next.val, curNode.node.next));
            }
        }
        return head.next;
    }


    List<Integer> path = new ArrayList<>();
    List<List<Integer>> res = new ArrayList<>();

    /**
     * 求一个整数数组的子集(二进制法)
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        int len = nums.length;
        for (int mask = 0; mask < (1 << len); mask++) {
            path.clear();
            for (int i = 0; i < len; i++) {
                if ((mask & (1 << i)) != 0) {
                    path.add(nums[i]);
                }
            }
            res.add(new ArrayList<>(path));
        }
        return res;
    }


    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


    /**
     * 判断一棵树是不是镜像对称二叉树
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    private boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }


    /**
     * 删除倒数第n个链表节点
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int index = 0;
        //哨兵节点
        ListNode res = new ListNode(0, head);
        ListNode fast = head;
        ListNode slow = res;
        while (index < n) {
            fast = fast.next;
            index++;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return res.next;
    }


    /**
     * 接雨水-栈求解
     *
     * @param height
     * @return
     */
    public static int trap(int[] height) {
        int res = 0;
        int len = height.length;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int max_width = i - stack.peek() - 1;
                int max_height = Math.min(height[i], height[stack.peek()]) - height[top];
                res += max_height * max_width;
            }
            stack.push(i);
        }
        return res;
    }

    /**
     * 接雨水-用双指针
     *
     * @param height
     * @return
     */
    public static int trap1(int[] height) {
        int left = 0, right = height.length - 1;
        int max_left = 0, max_right = 0;
        int res = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= max_left) {
                    max_left = height[left];
                } else {
                    res += (max_left - height[left]);
                }
                ++left;
            } else {
                if (height[right] >= max_right) {
                    max_right = height[right];
                } else {
                    res += (max_right - height[right]);
                }
                --right;
            }
        }
        return res;
    }

    /**
     * 删除有序数组中的重复项
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int p = 0;
        int q = 1;
        while (q < nums.length) {
            if (nums[p] != nums[q]) {
                if (q - p > 1) {
                    nums[p + 1] = nums[q];
                }
                p++;
            }
            q++;
        }
        return p + 1;
    }

    public static List<Integer> result = new ArrayList<>();

    /**
     * 分解成最小质因数的乘积
     *
     * @param n
     * @return
     */
    public static List primser(int n) {
        for (int i = 2; i <= n; i++) {
            if (n % i == 0) {
                result.add(i);
                primser(n / i);
                break;
            }
            if (i == n) {
                result.add(i);
            }
        }
        return result;
    }


    /**
     * 最长回文字串（动态规划）需要先列出状态转移方程
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || "".equals(s)) {
            return null;
        }
        if (s.length() < 2) {
            return s;
        }
        int maxLen = 1;
        int len = s.length();
        int begin = 0;
        boolean[][] dp = new boolean[len][len];
        //设置对角线为true
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] chars = s.toCharArray();
        for (int j = 1; j < len; j++) {
            for (int i = 0; i < j; i++) {
                if (chars[i] != chars[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && (j - i + 1) > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }


    /**
     * Z字形变换
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        String[] stringArr = new String[numRows];
        int period = numRows * 2 - 2;
        Arrays.fill(stringArr, "");
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            int mod = i % period;
            if (mod < numRows) {
                stringArr[mod] += chars[i];
            } else {
                stringArr[period - mod] += chars[i];
            }
        }
        StringBuffer stringBuffer = new StringBuffer();
        for (String subStr : stringArr) {
            stringBuffer.append(subStr);
        }
        return stringBuffer.toString();
    }


    /**
     * 盛最多水的容器
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int max = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                max = Math.max(max, height[left] * (right - left));
                left++;
            } else {
                max = Math.max(max, height[right] * (right - left));
                right--;
            }
        }
        return max;
    }


    /**
     * 柱状图中最大矩形
     *
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        Stack<Integer> stack = new Stack<>();
        //保存左右区间
        int[] left = new int[len];
        int[] right = new int[len];
        Arrays.fill(right, len);
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && heights[i] < heights[stack.peek()]) {
                right[stack.peek()] = i;
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
        }
        int res = 0;
        for (int i = 0; i < len; i++) {
            res = Math.max(res, heights[i] * (right[i] - left[i]));
        }
        return res;
    }


    /**
     * 最大矩形
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        if (m == 0) {
            return 0;
        }
        int n = matrix[0].length;
        int[][] left = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0 : left[i][j - 1]) + 1;
                }
            }
        }
        int ret = 0;
        for (int i = 0; i < n; i++) {
            int[] areaArr = new int[m];
            for (int j = 0; j < m; j++) {
                areaArr[j] = left[j][i];
            }
            ret = Math.max(ret, largestRectangleArea(areaArr));
        }
        return ret;
    }


    /**
     * 全排列(深度搜索+回溯法)
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        boolean[] used = new boolean[len];
        List<Integer> path = new ArrayList<>();

        dfs(nums, 0, len, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int depth, int len, List<Integer> path, boolean[] used, List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(nums[i]);
                used[i] = true;
                System.out.println("递归之前>>>>" + path);
                dfs(nums, depth + 1, len, path, used, res);
                used[i] = false;
                path.remove(path.size() - 1);
                System.out.println("递归之后>>>>" + path);
            }
        }
    }


    /**
     * 求二叉树的最大深度(BFS广度搜索)
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int count = 0;
        while (!queue.isEmpty()) {
            int index = 0;
            int size = queue.size();
            while (index < size) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                index++;
            }
            count++;
        }
        return count;
    }


    /**
     * 求二叉树的最大深度（递归法）
     *
     * @param root
     * @return
     */
    public int maxDepth1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth1(root.left), maxDepth1(root.right)) + 1;
    }


    /**
     * Dijkstra算法，求遍历图的最短路径（可用于有向图和无向图）
     *
     * @param arr 图的数组
     * @param n   顶点数
     * @return
     */
    public static int[] dijkstra(int[][] arr, int n) {
        //将图转化为邻接矩阵
        int[][] graph = new int[n][n];
        for (int i = 0; i < graph.length; i++) {
            //将无法到达的路径填充为-1
            Arrays.fill(graph[i], -1);
        }
        for (int[] child : arr) {
            graph[child[0] - 1][child[1] - 1] = child[2];
        }
        //记录顶点到下一个顶点的距离
        int[] result = new int[n];
        //记录当前点是否已经遍历
        boolean[] used = new boolean[n];
        //从顶点0开始遍历
        used[0] = true;
        System.out.print(1);
        for (int i = 0; i < graph.length; i++) {
            result[i] = graph[0][i];
        }
        //保存路径
        for (int i = 1; i < graph.length; i++) {
            int min = Integer.MAX_VALUE;
            int k = 0;
            for (int j = 1; j < graph.length; j++) {
                //找出下一个最短路径的节点
                if (!used[j] & (result[j] != -1) && (result[j] < min)) {
                    min = result[j];
                    k = j;
                }
            }
            used[k] = true;
            System.out.print("-->" + (k + 1));
            for (int j = 1; j < graph.length; j++) {
                if (!used[j]) {
                    //更新顶点到其他顶点的最短距离
                    if ((graph[k][j] != -1) && ((min + graph[k][j]) < result[j] || (result[j] == -1))) {
                        result[j] = min + graph[k][j];
                    }
                }
            }
        }
        int max = Integer.MIN_VALUE;
        for (int i : result) {
            max = Math.max(max, i);
        }
        System.out.println("\n" + "最短路径为：" + max);
        return result;
    }


    /**
     * 返回前k个出现次数最多的元素，最小堆揭发
     *
     * @param words
     * @param k
     * @return
     */
    public static List<String> topKFrequent(String[] words, int k) {
        List<String> result = new ArrayList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        PriorityQueue<Map.Entry<String, Integer>> priorityQueue = new PriorityQueue<>(
                (o1, o2) -> (o1.getValue() == o2.getValue() ? o2.getKey().compareTo(o1.getKey()) : o1.getValue() - o2.getValue()));
        map.entrySet().forEach(priorityQueue::offer);
        while (!priorityQueue.isEmpty()) {
            result.add(priorityQueue.poll().getKey());
        }
        int len = result.size() - k;
        for (int i = 0; i < len; i++) {
            result.remove(0);
        }
        Collections.reverse(result);
        return result;
    }


    /**
     * KMP算法
     *
     * @param pat
     * @param txt
     * @return
     */
    public static int KMPsearch(String pat, String txt) {
        int N = txt.length();
        int M = pat.length();
        for (int i = 0; i < N - M; i++) {
            int j = 0;
            for (; j < M; j++) {
                if (pat.charAt(j) != txt.charAt(i + j)) {
                    break;
                }
            }
            if (j == M) {
                return i;
            }
        }
        return -1;
    }


    /**
     * 括号生成
     *
     * @param n
     * @return
     */
    public static List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateALL(new char[n * 2], 0, result);
        return result;
    }

    private static void generateALL(char[] chars, int index, List<String> result) {
        if (index == chars.length) {
            if (checkParenthesis(chars)) {
                result.add(new String(chars));
            }
        } else {
            chars[index] = '(';
            generateALL(chars, index + 1, result);
            chars[index] = ')';
            generateALL(chars, index + 1, result);
        }
    }

    private static boolean checkParenthesis(char[] chars) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < chars.length; i++) {
            if (chars[i] == '(') {
                stack.push(chars[i]);
            } else if (chars[i] == ')') {
                if (stack.isEmpty()) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }


    /**
     * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     * <p>
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     * 使用二分法来实现
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = m * n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int cur = matrix[mid / n][mid % n];
            if (cur < target) {
                left = mid + 1;
            } else if (cur > target) {
                right = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }


    /**
     * 给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
     * <p>
     * 注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。
     * 回溯法超出内存
     *
     * @param s
     * @param words
     * @return
     */
    public static List<Integer> findSubstring(String s, String[] words) {
        HashSet set;
        int len = words.length;
        List<List<String>> res = new ArrayList<>();
        List<String> path = new ArrayList<>();
        boolean[] used = new boolean[len];
        findChildren(words, 0, len, path, used, res);
        List<String> transfer = new ArrayList<>();
        for (List<String> string : res) {
            transfer.add((String.join("", string)));
        }
        set = findChildrenPosition(s, transfer);
        return new ArrayList<>(set);
    }

    private static HashSet findChildrenPosition(String s, List<String> transfer) {
        int sLen = s.length();
        HashSet<Integer> set = new HashSet<>();
        for (String p : transfer) {
            int pLen = p.length();
            for (int i = 0; i <= (sLen - pLen); i++) {
                String str = s.substring(i, i + pLen);
                if (p.equals(str)) {
                    set.add(i);
                }
            }
        }
        return set;
    }

    public static void findChildren(String[] words, int depth, int len, List<String> path, boolean[] used, List<List<String>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(words[i]);
                used[i] = true;
                System.out.println("回溯前>>" + path);
                findChildren(words, depth + 1, len, path, used, res);
                used[i] = false;
                path.remove(path.size() - 1);
                System.out.println("回溯后>>" + path);
            }
        }
    }

    private static boolean searchChild(int i, int j, String s, String p) {
        System.out.println(i + "-" + j + "\t" + s + "\t" + p);
        String left = s.substring(0, i);
        System.out.println("left:" + left);
        String right = s.substring(j);
        System.out.println("right:" + right);
        System.out.println("===============================================");
        return (left.indexOf(p) != -1 || right.indexOf(p) != -1);

    }


    /**
     * 第76题：最小覆盖字串
     *
     * @param s
     * @param t
     * @return
     */
    public static String minWindow(String s, String t) {
        HashMap<Character, Integer> sMap = new HashMap<>();
        HashMap<Character, Integer> tMap = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            tMap.put(t.charAt(i), tMap.getOrDefault(t.charAt(i), 0) + 1);
        }
        String ans = "";
        int min = Integer.MAX_VALUE, cnt = 0;
        for (int i = 0, j = 0; i < s.length(); i++) {
            sMap.put(s.charAt(i), sMap.getOrDefault(s.charAt(i), 0) + 1);
            if (tMap.containsKey(s.charAt(i)) && sMap.get(s.charAt(i)) <= tMap.get(s.charAt(i))) {
                cnt++;
            }
            while (j < i && (!tMap.containsKey(s.charAt(j)) || sMap.get(s.charAt(j)) > tMap.get(s.charAt(j)))) {
                sMap.put(s.charAt(j), sMap.getOrDefault(s.charAt(j), 0) - 1);
                j++;
            }
            if (cnt == t.length() && i - j + 1 < min) {
                min = i - j + 1;
                ans = s.substring(j, i + 1);
            }
        }
        return ans;
    }


    private static boolean checkMap(HashMap<Character, Integer> sMap, HashMap<Character, Integer> tMap) {
        for (Map.Entry<Character, Integer> entry : tMap.entrySet()) {
            if (!sMap.containsKey(entry.getKey()) || sMap.get(entry.getKey()) < entry.getValue()) {
                return false;
            }
        }
        return true;
    }


    /**
     * 第56题：合并区间
     *
     * @param intervals
     * @return
     */
    public static int[][] merge(int[][] intervals) {
        int[][] res = new int[intervals.length][2];
        Arrays.sort(intervals, Comparator.comparingInt(v -> v[0]));
        int index = -1;
        for (int i = 0; i < intervals.length; i++) {
            if (index == -1 || intervals[i][0] > res[index][1]) {
                res[++index] = intervals[i];
            } else {
                res[index][1] = Math.max(res[index][1], intervals[i][1]);
            }
        }
        return Arrays.copyOf(res, index + 1);
    }


    /**
     * 第50题：Pow(x,n)
     *
     * @param x
     * @param n 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。
     * @return
     */
    public static double myPow(double x, int n) {
        return n >= 0 ? quickPow(x, n) : 1 / quickPow(x, -n);
    }

    private static double quickPow(double x, int n) {
        if (n == 0) {
            return 1.0;
        }
        double y = quickPow(x, n / 2);
        return n % 2 == 0 ? y * y : (y * y * x);
    }


    public static int search(int L, int a, long modulus, int n, int[] nums) {
        long h = 0;
        for (int i = 0; i < L; ++i) h = (h * a + nums[i]) % modulus;
        HashSet<Long> seen = new HashSet();
        seen.add(h);
        long aL = 1;
        for (int i = 1; i <= L; ++i) aL = (aL * a) % modulus;
        for (int start = 1; start < n - L + 1; ++start) {
            h = (h * a - nums[start - 1] * aL % modulus + modulus) % modulus;
            h = (h + nums[start + L - 1]) % modulus;
            if (seen.contains(h)) return start;
            seen.add(h);
        }
        return -1;
    }

    /**
     * 第1044题：寻找最长重复字串
     * 二分法？？？？？？？？？？？？？？？？？？？？？
     *
     * @param S
     */
    public static String longestDupSubstring(String S) {
        int n = S.length();
        int[] nums = new int[n];
        for (int i = 0; i < n; ++i) nums[i] = (int) S.charAt(i) - (int) 'a';
        int a = 26;
        long modulus = (long) Math.pow(2, 32);
        int left = 1, right = n;
        int L;
        while (left != right) {
            L = left + (right - left) / 2;
            if (search(L, a, modulus, n, nums) != -1) left = L + 1;
            else right = L;
        }
        int start = search(left - 1, a, modulus, n, nums);
        return start != -1 ? S.substring(start, start + left - 1) : "";
    }


    /**
     * 杨辉三角
     *
     * @param numRows
     * @return
     */
    public static List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < numRows; ++i) {
            List<Integer> t = new ArrayList<>();
            for (int j = 0; j < i + 1; ++j) {
                boolean firstOrLast = j == 0 || j == i;
                t.add(firstOrLast ? 1 : 0);
            }
            for (int j = 1; j < i; ++j) {
                int val = res.get(i - 1).get(j - 1) + res.get(i - 1).get(j);
                t.set(j, val);
            }
            res.add(t);
        }
        return res;
    }


    /**
     * 第1163题： 按字典序排在最后的子串
     * 1、找出所有的字串
     * 2、按字典排序，返回第一个
     *
     * @param s
     * @return
     */
    List<List<Character>> res1 = new ArrayList<>();
    List<Character> path1 = new ArrayList<>();

    public String lastSubstring1(String s) {
        char[] chars = s.toCharArray();
        int len = chars.length;
        for (int i = 0; i < (1 << len); i++) {
            path1.clear();
            for (int j = 0; j < len; j++) {
                if ((i & (1 << j)) != 0) {
                    path1.add(chars[j]);
                }
            }
            res1.add(new ArrayList<>(path1));
        }
        List<String> res = new ArrayList<>();
        for (List<Character> characters : res1) {
            StringBuffer sb = new StringBuffer();
            for (Character character : characters) {
                sb.append(character);
            }
            res.add(sb.toString());
        }
        res = res.stream().filter(e -> s.indexOf(e) != -1).sorted(Comparator.naturalOrder()).collect(Collectors.toList());
        return res.get(res.size() - 1);
    }


    /**
     * 第1163题：按字典序排在最后的字串
     *
     * @param s
     * @return
     */
    public String lastSubstring(String s) {
        int left = 0;
        int right = left + 1;
        int step = 0;
        while (right + step < s.length()) {
            if (s.charAt(left + step) < s.charAt(right + step)) {
                left = right;
                right++;
                step = 0;
            } else if (s.charAt(left + step) == s.charAt(right + step)) {
                step++;
            } else {
                right += step + 1;
                step = 0;
            }
        }
        return s.substring(left);
    }


    /**
     * 第10题：匹配正则表达式
     *
     * @param s
     * @param p
     * @return
     */
    public static boolean isMatch(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        int m = s.length();
        int n = p.length();
        //dp[i][j] 表示 s 的前 i 个是否能被 p 的前 j 个匹配
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i < n; i++) {
            if (p.charAt(i) == '*' && dp[0][i - 1]) {
                dp[0][i + 1] = true;
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
                    //如果是任意元素 或者是对于元素匹配
                    dp[i + 1][j + 1] = dp[i][j];
                }
                if (p.charAt(j) == '*') {
                    if (p.charAt(j - 1) != s.charAt(i) && p.charAt(j - 1) != '.') {
                        //如果前一个元素不匹配 且不为任意元素
                        dp[i + 1][j + 1] = dp[i + 1][j - 1];
                    } else {
                            /*
                            dp[i][j] = dp[i-1][j] // 多个字符匹配的情况
                            or dp[i][j] = dp[i][j-1] // 单个字符匹配的情况
                            or dp[i][j] = dp[i][j-2] // 没有匹配的情况
                             */
                        dp[i + 1][j + 1] = (dp[i][j + 1] || dp[i + 1][j] || dp[i + 1][j - 1]);
                    }
                }
            }
        }
        return dp[m][n];
    }


//    public TreeNode buildTree(int[] preorder, int[] inorder) {
//        int len = preorder.length;
//        HashMap<Integer, Integer> map = new HashMap<>();
//        //将中序结果保存在一个字典中
//        for (int i = 0; i < inorder.length; i++) {
//            map.put(inorder[i], i);
//        }
//        return build(preorder, 0, len - 1, inorder, 0, len - 1, map);
//
//    }
//
//    private TreeNode build(int[] preorder, int pre_left, int pre_right, int[] inorder, int in_left, int in_right, HashMap<Integer, Integer> map) {
//        if (pre_left > pre_right) {
//            return null;
//        }
//        //前序遍历结果根节点所在的位置为第一个，试想：假如换成后序遍历，则最后一个为根节点
//        int pre_root = 0;
//        //中序遍历结果中根节点所在的位置
//        int in_root = map.get(preorder[pre_root]);
//        //左子树的长度
//        int left_size = in_root-in_left;
//        TreeNode root = new TreeNode(preorder[pre_root]);
//        //找出前序和中序遍历结果中的左子树
//        root.left = build(preorder,pre_left+1,pre_left+left_size,inorder,in_right,left_size-1,map);
//        //找出前序和中序遍历结果中的右子树
//        root.right = build(preorder,pre_left+left_size + 1,pre_right,inorder,left_size+1,in_right,map);
//        return root;
//    }


    int[] preorder;
    HashMap<Integer, Integer> dic = new HashMap<>();

    /**
     * 构建二叉树
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for (int i = 0; i < inorder.length; i++)
            dic.put(inorder[i], i);
        return recur(0, 0, inorder.length - 1);
    }

    TreeNode recur(int root, int left, int right) {
        if (left > right) return null;                          // 递归终止
        TreeNode node = new TreeNode(preorder[root]);          // 建立根节点
        int i = dic.get(preorder[root]);                       // 划分根节点、左子树、右子树
        node.left = recur(root + 1, left, i - 1);              // 开启左子树递归
        node.right = recur(root + i - left + 1, i + 1, right); // 开启右子树递归
        return node;                                           // 回溯返回根节点
    }


    /**
     * 求最长递增子序列
     *
     * @param nums
     * @return
     */
    public static int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < dp.length; i++) {
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    public int midSearch(int[] nums, int target) {
        int left = 0;
        int right = nums.length;
        for (int i = 0; i < nums.length; i++) {
            int mid = left + (left + right) / 2;
            if (target == nums[mid]) {
                return i;
            } else if (target < nums[left]) {
                right = mid + 1;
            } else if (target > nums[right]) {
                left = mid - 1;
            }
        }
        return -1;
    }

    /**
     * 快速排序
     * @param arr
     * @param left
     * @param right
     * @return
     */
    public int[] quickSort(int[] arr, int left, int right) {
        if (left < right) {
            int partitionIndex = partition(arr, left, right);
            quickSort(arr, left, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, right);
        }
        return arr;
    }

    public int partition(int[] arr, int left, int right) {
        int pivot = left;
        int index = pivot + 1;
        for (int i = index; i <= right; i++) {
            if (arr[i] < arr[pivot]) {
                swap(arr, i, index);
                index++;
            }
        }
        swap(arr, pivot, index - 1);
        return index - 1;
    }

    private void swap(int[] arr, int a, int b) {
        int temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }

    /**
     * 第4题：寻找两个正序数组的中位数
     * @param A
     * @param B
     * @return
     */
    public double findMedianSortedArrays1(int[] A, int[] B) {
        int m = A.length;
        int n = B.length;
        int len = m + n;
        int left = -1, right = -1;
        int aStart = 0, bStart = 0;
        for (int i = 0; i <= len / 2; i++) {
            left = right;
            if (aStart < m && (bStart >= n || A[aStart] < B[bStart])) {
                right = A[aStart++];
            } else {
                right = B[bStart++];
            }
        }
        if ((len & 1) == 0)
            return (left + right) / 2.0;
        else
            return right;
    }

    /**
     * 第4题：寻找两个正序数组的中位数
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */
        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;

        while (true) {
            // 边界情况
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            // 正常情况
            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
//        System.out.println(lengthOfLIS(new int[]{1,5,7,9,2,4,6,7,4,6}));
//        System.out.println("kangbinqiang".indexOf("kang") != -1);
//        System.out.println(solution.lastSubstring("vmjtxddvzmwrjvfamgpoowncslddrkjhchqswkamnsitrcmnhn"));
//        System.out.println(longestDupSubstring("kangbinqiangbinqiang"));
//        System.out.println(generate(6));
//        System.out.println(merge(new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}}));
//        System.out.println(minWindow("ADOBECODEBANC", "ABC"));
//        String[] strs = new String[]{"hello","hello world","hello China"};
//        System.out.println(longestCommonPrefix(strs));
//        System.out.println(invalidTransactions(new String[]{"bob,689,1910,barcelona", "alex,696,122,bangkok", "bob,832,1726,barcelona", "bob,820,596,bangkok", "chalicefy,217,669,barcelona", "bob,175,221,amsterdam"}));
//        System.out.println(trap1(new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}));
//        System.out.println(primser(12));
//        int[] nums = {1, 2, 3};
//        Solution solution = new Solution();
//        List<List<Integer>> lists = solution.permute(nums);
//        int[][] graph = new int[][]{{1, 3, 10}, {1, 5, 30}, {1, 6, 100}, {2, 3, 5}, {3, 4, 50}, {4, 6, 10}, {5, 6, 60}, {5, 4, 20}};
//        dijkstra(graph, 6);
//        System.out.println(dijkstra(graph,6));
//        String[] graph = new String[]{"the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"};
//        System.out.println(topKFrequent(graph, 2));
//        System.out.println(KMPsearch("i","kangbinqiang"));

//        System.out.println(generateParenthesis(3));
//        System.out.println(longestDupSubstring("hello"));
//        System.out.println(findSubstring("foobarffsfsdfsdfsdfsdfsfsdfsoobar", new String[]{"kang", "bin", "qiang"}));
//        System.out.println(generateParenthesis(4));
        System.out.println(solution.quickSort(new int[]{3,67,4,7,8,23,54,9},0,7));
        
    }


}
