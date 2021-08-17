package leetcode;

import sun.awt.windows.WPrinterJob;

import java.util.*;

public class Solution {


    /**
     * 鑾峰彇瀛楃涓叉渶闀跨殑鍏叡鍓嶇紑
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            prefix = longestCommonPrefix(prefix, strs[i]);
            if (prefix.length() == 0) {
                break;
            }
        }
        return prefix;
    }

    private String longestCommonPrefix(String str1, String str2) {
        int length = Math.min(str1.length(), str2.length());
        int index = 0;
        while (index < length && str1.charAt(index) == str2.charAt(index)) {
            index++;
        }
        return str1.substring(0, index);
    }


    class ListNode {
        int val;
        ListNode next;

        public ListNode(int val) {
            this.val = val;
        }
    }

    /**
     * 鍚堝苟涓や釜鏈夊簭閾捐〃
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        } else if (l1.val <= l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /**
     * 鍒犻櫎閾捐〃鐨勫�掓暟绗琻涓妭鐐�
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode start = pre, end = pre;
        while (n != 0) {
            start = start.next;
            n--;
        }
        while (start.next != null) {
            start = start.next;
            end = end.next;
        }
        end.next = end.next.next;
        return pre.next;
    }


//    /**
//     * 姹傛渶闀跨殑鍥炴枃瀛愪覆
//     *
//     * @param s
//     * @return
//     */
//    public String longestPalindrome(String s) {
//        if (s == null || "".equals(s)) {
//            return null;
//        }
//        int max = 0;
//        int left = 0;
//        for (int i = 0; i < s.length(); i++) {
//            for (int j = 0; j < s.length(); j++) {
//                if (j - i + 1 > max && isPalindrome(s, i, j)) {
//                    max = j - i + 1;
//                    left = i;
//                }
//            }
//        }
//        return s.substring(left, left + max);
//    }
//
//    private boolean isPalindrome(String s, int i, int j) {
//        char[] chars = s.toCharArray();
//        while (i < j) {
//            if (chars[i] != chars[j]) {
//                return false;
//            }
//            i++;
//            j--;
//        }
//        return true;
//    }


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
        //璁剧疆瀵硅绾夸负true
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
     * 灏嗕竴涓粰瀹氬瓧绗︿覆 s 鏍规嵁缁欏畾鐨勮鏁� numRows 锛屼互浠庝笂寰�涓嬨�佷粠宸﹀埌鍙宠繘琛�?Z 瀛楀舰鎺掑垪銆�
     * <p>
     * 姣斿杈撳叆瀛楃涓蹭负 "PAYPALISHIRING"?琛屾暟涓� 3 鏃讹紝鎺掑垪濡備笅锛�
     * <p>
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     * 涔嬪悗锛屼綘鐨勮緭鍑洪渶瑕佷粠宸﹀線鍙抽�愯璇诲彇锛屼骇鐢熷嚭涓�涓柊鐨勫瓧绗︿覆锛屾瘮濡傦細"PAHNAPLSIIGYIR"銆�
     * <p>
     * 璇蜂綘瀹炵幇杩欎釜灏嗗瓧绗︿覆杩涜鎸囧畾琛屾暟鍙樻崲鐨勫嚱鏁帮細
     * <p>
     * string convert(string s, int numRows);
     * ?
     * <p>
     * 绀轰緥 1锛�
     * <p>
     * 杈撳叆锛歴 = "PAYPALISHIRING", numRows = 3
     * 杈撳嚭锛�"PAHNAPLSIIGYIR"
     * 绀轰緥 2锛�
     * 杈撳叆锛歴 = "PAYPALISHIRING", numRows = 4
     * 杈撳嚭锛�"PINALSIGYAHRPI"
     * 瑙ｉ噴锛�
     * P     I    N
     * A   L S  I G
     * Y A   H R
     * P     I
     * 绀轰緥 3锛�
     * <p>
     * 杈撳叆锛歴 = "A", numRows = 1
     * 杈撳嚭锛�"A"
     *
     * @param s
     * @param numRows
     * @return
     */
    public static String convert(String s, int numRows) {
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
     * 鐩涙渶澶氭按鐨勫鍣�
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
     * 璁＄畻涓�缁存暟缁勬渶澶х煩褰㈤潰绉�
     *
     * @param heights
     * @return
     */
    public static int largestRectangleArea(int[] heights) {
        int len = heights.length;
        Stack<Integer> stack = new Stack<>();
        //淇濆瓨宸﹀彸鍖洪棿
        int[] left = new int[len];
        int[] right = new int[len];
        Arrays.fill(right, len);
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                right[stack.peek()] = i;
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        int res = 0;
        for (int i = 0; i < len; i++) {
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        return res;
    }


    /**
     * 璁＄畻浜岀淮鏁扮粍鐨勬渶澶х煩褰㈤潰绉�
     *
     * @param matrix
     * @return
     */
    public static int maximalRectangle(char[][] matrix) {
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
     * 姹傚浘鐨勬渶鐭矾寰勶紙Dijkstra锛�
     *
     * @param times
     * @param n
     * @param k
     * @return
     */
    public static int networkDelayTime(int[][] times, int n, int k) {
//        System.out.print(k - 1);
        int max = Integer.MIN_VALUE;
        //鏋勫缓鐭╅樀
        int[] road = new int[n];
        int count = 1;
        int[][] adjMatrix = new int[n][n];
        for (int[] matrix : adjMatrix) {
            Arrays.fill(matrix, -1);
        }
        for (int[] time : times) {
            adjMatrix[time[0] - 1][time[1] - 1] = time[2];
        }
        //S鏁扮粍
        int[] result = new int[adjMatrix.length];
        boolean[] used = new boolean[adjMatrix.length];
        //鏍囪瘑k浣滀负椤剁偣
        used[k - 1] = true;
        road[0] = k - 1;
        for (int i = 0; i < adjMatrix.length; i++) {
            result[i] = adjMatrix[k - 1][i];
        }
        for (int i = 0; i < adjMatrix.length; i++) {
            if (i == (k - 1)) {
                continue;
            }
            int min = Integer.MAX_VALUE;
            int cur = 0;
            for (int j = 0; j < adjMatrix.length; j++) {
                //绗琷涓繕娌℃湁閬嶅巻鍒�
                if (!used[j] && result[j] != -1 && result[j] < min) {
                    min = result[j];
                    cur = j;
                }
            }
//            System.out.print("--->" + cur);
            road[count] = cur;
            count++;
            //鏇存柊U鏁扮粍
            used[cur] = true;
            for (int j = 0; j < adjMatrix.length; j++) {
                if (!used[j]) {
                    if (adjMatrix[cur][j] != -1 && (result[j] > min + adjMatrix[cur][j] || result[j] == -1)) {
                        result[j] = min + adjMatrix[cur][j];
                    }
                }
            }

        }
        Set<String> set = new HashSet<>();
        for (int[] time : times) {
            set.add(String.valueOf(time[0] - 1) + '-' + (time[1] - 1));
        }
        for (int i = 0; i < road.length - 1; i++) {
            if (!set.contains(String.valueOf(road[i]) + '-' + road[i + 1])) {
                return -1;
            }
        }
        for (int i : result) {
            max = Math.max(max, i);
        }
//        System.out.println("\n");
        return max > Integer.MAX_VALUE / 2 ? -1 : max;
    }


    public static int networkDelayTime1(int[][] times, int n, int k) {
        final int INF = Integer.MAX_VALUE / 2;
        int[][] g = new int[n][n];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(g[i], INF);
        }
        for (int[] t : times) {
            g[t[0] - 1][t[1] - 1] = t[2];
        }

        int[] dist = new int[n];
        Arrays.fill(dist, INF);
        dist[k - 1] = 0;
        boolean[] used = new boolean[n];
        for (int i = 0; i < n; ++i) {
            int x = -1;
            for (int y = 0; y < n; ++y) {
                if (!used[y] && (x == -1 || dist[y] < dist[x])) {
                    x = y;
                }
            }
            System.out.print("-->" + (x + 1));
            used[x] = true;
            for (int y = 0; y < n; ++y) {
                dist[y] = Math.min(dist[y], dist[x] + g[x][y]);
            }
        }

        int ans = Arrays.stream(dist).max().getAsInt();
        System.out.println("\n");
        return ans == INF ? -1 : ans;
    }


    /**
     * 鎵旈浮铔� 閫掑綊瑙ｆ硶dp[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
     *
     * @param k
     * @param n
     * @return
     */
    HashMap<String, Integer> map = new HashMap<>();

    public int superEggDrop1(int k, int n) {

        if (k == 1) {
            return n;
        }
        if (n == 0) {
            return 0;
        }
        if (map.containsKey(k + "-" + n)) {
            return map.get(k + "-" + n);
        }
        int res = n;
        for (int i = 1; i <= n; i++) {
            int maxFloor = Math.max(superEggDrop(k, n - i), superEggDrop(k - 1, i - 1)) + 1;
            res = Math.min(res, maxFloor);
        }
        map.put(k + "-" + n, res);

        return res;
    }


    /**
     * 鍔ㄦ�佽鍒掕В娉昫p[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
     *
     * @param k
     * @param n
     * @return
     */
    public int superEggDrop(int k, int n) {
        int dp[][] = new int[k + 1][n + 1];
        //鍙湁涓�涓浮铔嬬殑鏃跺�欙紝鏈�澶氶渶瑕乶娆�;娌℃湁楦¤泲鐨勬椂鍊欙紝閮芥槸0娆�
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 0;
            dp[1][i] = i;
        }
        //鍙湁0灞傜殑鏃跺�欙紝閮芥槸0娆�
        for (int i = 0; i <= k; i++) {
            dp[i][0] = 0;
        }
        for (int i = 2; i <= k; i++) {
            for (int j = 1; j <= n; j++) {
                int min = Integer.MAX_VALUE;
                for (int x = 1; x <= j; x++) {
                    min = Math.min(min, Math.max(dp[i - 1][x - 1], dp[i][j - x]) + 1);
                }
                dp[i][j] = min;
            }
        }
        return dp[k][n];
    }

    /**
     * 鏂愭尝閭ｅ鏁板垪鎸囬拡瑙ｆ硶
     *
     * @param n
     * @return
     */
    public static int fib1(int n) {
        if (n < 2) {
            return n;
        }
        int p = 0, q = 0, r = 1;
        for (int i = 2; i <= n; i++) {
            p = q;
            q = r;
            r = p + q;
        }
        return r;
    }

    /**
     * 鏂愭尝閭ｅ鍔ㄦ�佽鍒掕В娉�
     *
     * @param n
     * @return
     */
    public static int fib(int n) {
        if (n < 1) {
            return n;
        }
        int[] arr = new int[n];
        return fn(arr, n);
    }

    private static int fn(int[] arr, int n) {
        if (n == 1 || n == 2) {
            return 1;
        }
        if (arr[n - 1] != 0) {
            return arr[n - 1];
        }
        arr[n - 1] = fn(arr, n - 1) + fn(arr, n - 2);
        return arr[n - 1];
    }


    /**
     * 鎷彿鐢熸垚锛堟毚鍔涜В娉曪級
     *
     * @param n
     * @return
     */
    public static List<String> generateParenthesis1(int n) {
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
     * 鎷彿鐢熸垚锛堝洖婧硶锛�
     *
     * @param n
     * @return
     */
    public static List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        backtrack(new StringBuffer(), res, 0, 0, n);
        return res;
    }

    private static void backtrack(StringBuffer cur, List<String> res, int left, int right, int n) {
        if (cur.length() == 2 * n) {
            res.add(cur.toString());
            return;
        }
        if (left < n) {
            cur.append('(');
            backtrack(cur, res, left + 1, right, n);
            cur.deleteCharAt(cur.length() - 1);
        }
        if (right < left) {
            cur.append(')');
            backtrack(cur, res, left, right + 1, n);
            cur.deleteCharAt(cur.length() - 1);
        }
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
     * 浜屽弶鏍戝厛搴忛亶鍘�(閫掑綊娉�)
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal1(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        preOrder(root, res);
        return res;
    }

    private void preOrder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        res.add(root.val);
        preOrder(root.left, res);
        preOrder(root.right, res);
    }


    /**
     * 浜屽弶鏍戠殑鍏堝簭閬嶅巻锛堟暟鎹粨鏋勬硶锛�
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                res.add(node.val);
                stack.push(node);
            }
            node = stack.pop();
            node = node.right;
        }
        return res;
    }

    /**
     * 浜屽弶鏍戝睍寮�涓洪摼琛�
     *
     * @param root
     */
    public void flatten(TreeNode root) {

    }


    /**
     * 杩斿洖鍓峩涓嚭鐜版鏁版渶澶氱殑鍏冪礌锛屾渶灏忓爢鎻彂
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


    public static HashMap<Integer, Integer> treeMap = new HashMap<>();

    /**
     * 给定二叉树的前序遍历和中序遍历结果，生成二叉树
     * Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
     * Output: [3,9,20,null,null,15,7]
     * 1、首先二叉树的前序遍历结果中，第一个节点就是根节点
     * 2、在二叉树中序遍历结果中找出根节点所在的下标，则左边的为左子树，右边的为右子树
     * 3、每一个左子树序列又能在前序中找出根节点，依次循环1、2、3即可完成二叉树的创建
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = preorder.length;

        //将中序结果保存在一个字典中
        for (int i = 0; i < inorder.length; i++) {
            treeMap.put(inorder[i], i);
        }
        return build(preorder, 0, len - 1, inorder, 0, len - 1);

    }

    public TreeNode build(int[] preorder, int pre_left, int pre_right, int[] inorder, int in_left, int in_right) {
        if (pre_left > pre_right) {
            return null;
        }
        //前序遍历结果根节点所在的位置为第一个，试想：假如换成后序遍历，则最后一个为根节点
        int pre_root = 0;
        //中序遍历结果中根节点所在的位置
        int in_root = treeMap.get(preorder[pre_root]);
        //左子树的长度
        int left_size = in_root - in_left;
        TreeNode root = new TreeNode(preorder[pre_root]);
        //找出前序和中序遍历结果中的左子树
        root.left = build(preorder, pre_left + 1, pre_left + left_size, inorder, in_right, left_size - 1);
        //找出前序和中序遍历结果中的右子树
        root.right = build(preorder, pre_left + left_size + 1, pre_right, inorder, left_size + 1, in_right);
        return root;
    }


    /**
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int flag = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + flag;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            flag = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (flag > 0) {
            tail.next = new ListNode(flag);
        }
        return head;
    }


    /**
     * 寻找最长重复子串
     * 1、递归找出所有的子串
     * 2、KMP算法
     *
     * @param s
     * @return
     */
    public static String longestDupSubstring(String s) {
        int max = Integer.MIN_VALUE;
        String result = "";
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) {
                String p = s.substring(i, j);
                System.out.println(p);
                //此时以子串将源串分为左右两个，假如任何一个中存在子串，则返回结果
                if (searchChild(i, j, s, p)) {
                    if ((j - i) > max) {
                        max = j - i;
                        result = s.substring(i, j);
                    }
                }
            }
        }
        return result;
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


    public static void main(String[] args) {
        System.out.println(longestDupSubstring("banana"));
        ;
//        int[][] times = new int[][]{{2, 1, 1}, {2, 3, 1}, {3, 4, 1}};
//        int n = 4;
//        int k = 2;
//        System.out.println(networkDelayTime1(times, n, k));
//        System.out.println("hello world");
//        System.out.println(convert("PAYPALISHIRING", 4));
//        char[][] arr = new char[][]{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}};
//        System.out.println(maximalRectangle(arr));7
//        for (int i = 0; i < 31; i++) {
//            System.out.println(fib(i));
//        }
//        System.out.println(generateParenthesis(3));
    }

}
