package leetcode;

import java.util.*;

public class Solution {


    /**
     * ��ȡ�ַ�����Ĺ���ǰ׺
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
     * �ϲ�������������
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
     * ɾ������ĵ�����n���ڵ�
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
//     * ����Ļ����Ӵ�
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
        //���öԽ���Ϊtrue
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
     * ��һ�������ַ��� s ���ݸ��������� numRows ���Դ������¡������ҽ���?Z �������С�
     * <p>
     * ���������ַ���Ϊ "PAYPALISHIRING"?����Ϊ 3 ʱ���������£�
     * <p>
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     * ֮����������Ҫ�����������ж�ȡ��������һ���µ��ַ��������磺"PAHNAPLSIIGYIR"��
     * <p>
     * ����ʵ��������ַ�������ָ�������任�ĺ�����
     * <p>
     * string convert(string s, int numRows);
     * ?
     * <p>
     * ʾ�� 1��
     * <p>
     * ���룺s = "PAYPALISHIRING", numRows = 3
     * �����"PAHNAPLSIIGYIR"
     * ʾ�� 2��
     * ���룺s = "PAYPALISHIRING", numRows = 4
     * �����"PINALSIGYAHRPI"
     * ���ͣ�
     * P     I    N
     * A   L S  I G
     * Y A   H R
     * P     I
     * ʾ�� 3��
     * <p>
     * ���룺s = "A", numRows = 1
     * �����"A"
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
     * ʢ���ˮ������
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
     * ����һά�������������
     *
     * @param heights
     * @return
     */
    public static int largestRectangleArea(int[] heights) {
        int len = heights.length;
        Stack<Integer> stack = new Stack<>();
        //������������
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
     * �����ά��������������
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
     * ��ͼ�����·����Dijkstra��
     *
     * @param times
     * @param n
     * @param k
     * @return
     */
    public static int networkDelayTime(int[][] times, int n, int k) {
//        System.out.print(k - 1);
        int max = Integer.MIN_VALUE;
        //��������
        int[] road = new int[n];
        int count = 1;
        int[][] adjMatrix = new int[n][n];
        for (int[] matrix : adjMatrix) {
            Arrays.fill(matrix, -1);
        }
        for (int[] time : times) {
            adjMatrix[time[0] - 1][time[1] - 1] = time[2];
        }
        //S����
        int[] result = new int[adjMatrix.length];
        boolean[] used = new boolean[adjMatrix.length];
        //��ʶk��Ϊ����
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
                //��j����û�б�����
                if (!used[j] && result[j] != -1 && result[j] < min) {
                    min = result[j];
                    cur = j;
                }
            }
//            System.out.print("--->" + cur);
            road[count] = cur;
            count++;
            //����U����
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
     * �Ӽ��� �ݹ�ⷨdp[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
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
     * ��̬�滮�ⷨdp[k][n] = min(max(dp[k-1][i-1],dp[k][n-i])+1)
     *
     * @param k
     * @param n
     * @return
     */
    public int superEggDrop(int k, int n) {
        int dp[][] = new int[k + 1][n + 1];
        //ֻ��һ��������ʱ�������Ҫn��;û�м�����ʱ�򣬶���0��
        for (int i = 0; i <= n; i++) {
            dp[0][i] = 0;
            dp[1][i] = i;
        }
        //ֻ��0���ʱ�򣬶���0��
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
     * 쳲���������ָ��ⷨ
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
     * 쳲�������̬�滮�ⷨ
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
     * �������ɣ������ⷨ��
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
            generateALL(chars,index+1,result);
            chars[index] = ')';
            generateALL(chars,index+1,result);
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
     * �������ɣ����ݷ���
     * @param n
     * @return
     */
    public static List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        backtrack(new StringBuffer(),res,0,0,n);
        return res;
    }

    private static void backtrack(StringBuffer cur, List<String> res, int left, int right, int n) {
        if (cur.length() == 2 * n) {
            res.add(cur.toString());
            return;
        }
        if (left < n) {
            cur.append('(');
            backtrack(cur,res,left+1,right,n);
            cur.deleteCharAt(cur.length()-1);
        }
        if (right < left) {
            cur.append(')');
            backtrack(cur,res,left,right+1,n);
            cur.deleteCharAt(cur.length()-1);
        }
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
