package leetcode;

import java.util.*;

public class Solution {


    /**
     * è·å–å­—ç¬¦ä¸²æœ€é•¿çš„å…¬å…±å‰ç¼€
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
     * åˆå¹¶ä¸¤ä¸ªæœ‰åºé“¾è¡¨
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
     * åˆ é™¤é“¾è¡¨çš„å€’æ•°ç¬¬nä¸ªèŠ‚ç‚¹
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
//     * æ±‚æœ€é•¿çš„å›æ–‡å­ä¸²
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
        //è®¾ç½®å¯¹è§’çº¿ä¸ºtrue
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
     * å°†ä¸€ä¸ªç»™å®šå­—ç¬¦ä¸² s æ ¹æ®ç»™å®šçš„è¡Œæ•° numRows ï¼Œä»¥ä»ä¸Šå¾€ä¸‹ã€ä»å·¦åˆ°å³è¿›è¡Œ?Z å­—å½¢æ’åˆ—ã€‚
     * <p>
     * æ¯”å¦‚è¾“å…¥å­—ç¬¦ä¸²ä¸º "PAYPALISHIRING"?è¡Œæ•°ä¸º 3 æ—¶ï¼Œæ’åˆ—å¦‚ä¸‹ï¼š
     * <p>
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     * ä¹‹åï¼Œä½ çš„è¾“å‡ºéœ€è¦ä»å·¦å¾€å³é€è¡Œè¯»å–ï¼Œäº§ç”Ÿå‡ºä¸€ä¸ªæ–°çš„å­—ç¬¦ä¸²ï¼Œæ¯”å¦‚ï¼š"PAHNAPLSIIGYIR"ã€‚
     * <p>
     * è¯·ä½ å®ç°è¿™ä¸ªå°†å­—ç¬¦ä¸²è¿›è¡ŒæŒ‡å®šè¡Œæ•°å˜æ¢çš„å‡½æ•°ï¼š
     * <p>
     * string convert(string s, int numRows);
     * ?
     * <p>
     * ç¤ºä¾‹ 1ï¼š
     * <p>
     * è¾“å…¥ï¼šs = "PAYPALISHIRING", numRows = 3
     * è¾“å‡ºï¼š"PAHNAPLSIIGYIR"
     * ç¤ºä¾‹ 2ï¼š
     * è¾“å…¥ï¼šs = "PAYPALISHIRING", numRows = 4
     * è¾“å‡ºï¼š"PINALSIGYAHRPI"
     * è§£é‡Šï¼š
     * P     I    N
     * A   L S  I G
     * Y A   H R
     * P     I
     * ç¤ºä¾‹ 3ï¼š
     * <p>
     * è¾“å…¥ï¼šs = "A", numRows = 1
     * è¾“å‡ºï¼š"A"
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
     * ç››æœ€å¤šæ°´çš„å®¹å™¨
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
     * è®¡ç®—ä¸€ç»´æ•°ç»„æœ€å¤§çŸ©å½¢é¢ç§¯
     *
     * @param heights
     * @return
     */
    public static int largestRectangleArea(int[] heights) {
        int len = heights.length;
        Stack<Integer> stack = new Stack<>();
        //ä¿å­˜å·¦å³åŒºé—´
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
     * è®¡ç®—äºŒç»´æ•°ç»„çš„æœ€å¤§çŸ©å½¢é¢ç§¯
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
     * æ±‚å›¾çš„æœ€çŸ­è·¯å¾„ï¼ˆDijkstraï¼‰
     *
     * @param times
     * @param n
     * @param k
     * @return
     */
    public static int networkDelayTime(int[][] times, int n, int k) {
//        System.out.print(k - 1);
        int max = Integer.MIN_VALUE;
        //æ„å»ºçŸ©é˜µ
        int[] road = new int[n];
        int count = 1;
        int[][] adjMatrix = new int[n][n];
        for (int[] matrix : adjMatrix) {
            Arrays.fill(matrix, -1);
        }
        for (int[] time : times) {
            adjMatrix[time[0] - 1][time[1] - 1] = time[2];
        }
        //Sæ•°ç»„
        int[] result = new int[adjMatrix.length];
        boolean[] used = new boolean[adjMatrix.length];
        //æ ‡è¯†kä½œä¸ºé¡¶ç‚¹
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
                //ç¬¬jä¸ªè¿˜æ²¡æœ‰éå†åˆ°
                if (!used[j] && result[j] != -1 && result[j] < min) {
                    min = result[j];
                    cur = j;
                }
            }
//            System.out.print("--->" + cur);
            road[count] = cur;
            count++;
            //æ›´æ–°Uæ•°ç»„
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
     * æ‰”é¸¡è›‹ é€’å½’è§£æ³•dp[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
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
     * åŠ¨æ€è§„åˆ’è§£æ³•dp[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
     *
     * @param k
     * @param n
     * @return
     */
    public int superEggDrop(int k, int n) {
        int dp[][] = new int[k + 1][n + 1];
        //åªæœ‰ä¸€ä¸ªé¸¡è›‹çš„æ—¶å€™ï¼Œæœ€å¤šéœ€è¦næ¬¡;æ²¡æœ‰é¸¡è›‹çš„æ—¶å€™ï¼Œéƒ½æ˜¯0æ¬¡
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 0;
            dp[1][i] = i;
        }
        //åªæœ‰0å±‚çš„æ—¶å€™ï¼Œéƒ½æ˜¯0æ¬¡
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
     * æ–æ³¢é‚£å¥‘æ•°åˆ—æŒ‡é’ˆè§£æ³•
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
     * æ–æ³¢é‚£å¥‘åŠ¨æ€è§„åˆ’è§£æ³•
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
     * æ‹¬å·ç”Ÿæˆï¼ˆæš´åŠ›è§£æ³•ï¼‰
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
     * æ‹¬å·ç”Ÿæˆï¼ˆå›æº¯æ³•ï¼‰
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
     * äºŒå‰æ ‘å…ˆåºéå†(é€’å½’æ³•)
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
     * äºŒå‰æ ‘çš„å…ˆåºéå†ï¼ˆæ•°æ®ç»“æ„æ³•ï¼‰
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
     * äºŒå‰æ ‘å±•å¼€ä¸ºé“¾è¡¨
     *
     * @param root
     */
    public void flatten(TreeNode root) {

    }
    
    
     /**
     * è¿”å›å‰kä¸ªå‡ºç°æ¬¡æ•°æœ€å¤šçš„å…ƒç´ ï¼Œæœ€å°å †æ­å‘
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
        int len = result.size()-k;
        for (int i = 0; i < len; i++) {
            result.remove(0);
        }
        Collections.reverse(result);
        return result;
    }


    public static HashMap<Integer, Integer> treeMap = new HashMap<>();

    /**
     * ¸ø¶¨¶ş²æÊ÷µÄÇ°Ğò±éÀúºÍÖĞĞò±éÀú½á¹û£¬Éú³É¶ş²æÊ÷
     * Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
     * Output: [3,9,20,null,null,15,7]
     * 1¡¢Ê×ÏÈ¶ş²æÊ÷µÄÇ°Ğò±éÀú½á¹ûÖĞ£¬µÚÒ»¸ö½Úµã¾ÍÊÇ¸ù½Úµã
     * 2¡¢ÔÚ¶ş²æÊ÷ÖĞĞò±éÀú½á¹ûÖĞÕÒ³ö¸ù½ÚµãËùÔÚµÄÏÂ±ê£¬Ôò×ó±ßµÄÎª×ó×ÓÊ÷£¬ÓÒ±ßµÄÎªÓÒ×ÓÊ÷
     * 3¡¢Ã¿Ò»¸ö×ó×ÓÊ÷ĞòÁĞÓÖÄÜÔÚÇ°ĞòÖĞÕÒ³ö¸ù½Úµã£¬ÒÀ´ÎÑ­»·1¡¢2¡¢3¼´¿ÉÍê³É¶ş²æÊ÷µÄ´´½¨
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = preorder.length;

        //½«ÖĞĞò½á¹û±£´æÔÚÒ»¸ö×ÖµäÖĞ
        for (int i = 0; i < inorder.length; i++) {
            treeMap.put(inorder[i], i);
        }
        return build(preorder, 0, len - 1, inorder, 0, len - 1);

    }

    public TreeNode build(int[] preorder, int pre_left, int pre_right, int[] inorder, int in_left, int in_right) {
        if (pre_left > pre_right) {
            return null;
        }
        //Ç°Ğò±éÀú½á¹û¸ù½ÚµãËùÔÚµÄÎ»ÖÃÎªµÚÒ»¸ö£¬ÊÔÏë£º¼ÙÈç»»³ÉºóĞò±éÀú£¬Ôò×îºóÒ»¸öÎª¸ù½Úµã
        int pre_root = 0;
        //ÖĞĞò±éÀú½á¹ûÖĞ¸ù½ÚµãËùÔÚµÄÎ»ÖÃ
        int in_root = treeMap.get(preorder[pre_root]);
        //×ó×ÓÊ÷µÄ³¤¶È
        int left_size = in_root - in_left;
        TreeNode root = new TreeNode(preorder[pre_root]);
        //ÕÒ³öÇ°ĞòºÍÖĞĞò±éÀú½á¹ûÖĞµÄ×ó×ÓÊ÷
        root.left = build(preorder, pre_left + 1, pre_left + left_size, inorder, in_right, left_size - 1);
        //ÕÒ³öÇ°ĞòºÍÖĞĞò±éÀú½á¹ûÖĞµÄÓÒ×ÓÊ÷
        root.right = build(preorder, pre_left + left_size + 1, pre_right, inorder, left_size + 1, in_right);
        return root;
    }


    /**
     * ¸øÄãÁ½¸ö ·Ç¿Õ µÄÁ´±í£¬±íÊ¾Á½¸ö·Ç¸ºµÄÕûÊı¡£ËüÃÇÃ¿Î»Êı×Ö¶¼ÊÇ°´ÕÕ ÄæĞò µÄ·½Ê½´æ´¢µÄ£¬²¢ÇÒÃ¿¸ö½ÚµãÖ»ÄÜ´æ´¢ Ò»Î» Êı×Ö¡£
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


    public static void main(String[] args) {
//        int[][] times = new int[][]{{2, 1, 1}, {2, 3, 1}, {3, 4, 1}};
//        int n = 4;
//        int k = 2;
//        System.out.println(networkDelayTime1(times, n, k));
//        System.out.println("hello world");
//        System.out.println(convert("PAYPALISHIRING", 4));
//        char[][] arr = new char[][]{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}};
//        System.out.println(maximalRectangle(arr));
//        for (int i = 0; i < 31; i++) {
//            System.out.println(fib(i));
//        }
        System.out.println(generateParenthesis(3));
    }

}
