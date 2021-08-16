package leetcode;

/**
 * 循环队列
 */
class MyCircularQueue {

    int capacity;
    int headIndex;
    int[] queue;
    int curSize;

    /**
     * 初始化队列
     *
     * @param k
     */
    public MyCircularQueue(int k) {
        this.queue = new int[k];
        this.capacity = k;
        this.headIndex = 0;
        this.curSize = 0;
    }

    /**
     * 进队列
     * 1、队列是否已经满了？
     * 2、修改队列的当前长度
     *
     * @param value
     * @return
     */
    public boolean enQueue(int value) {
        if (isFull()) {
            return false;
        }
        this.queue[(headIndex + curSize) % this.capacity] = value;
        this.curSize++;
        return true;

    }

    /**
     * 队首出队列
     * @return
     */
    public boolean deQueue() {
        if (isEmpty()) {
            return false;
        }
        this.headIndex = (this.headIndex + 1) % this.capacity;
        this.curSize--;
        return true;
    }

    /**
     * 返回队首
     * @return
     */
    public int Front() {
        if (isEmpty()) {
            return -1;
        }
        return this.queue[this.headIndex];
    }

    /**
     * 返回队尾
     * @return
     */
    public int Rear() {
        if (isEmpty()) {
            return -1;
        }
        return this.queue[(this.headIndex + this.curSize -1) % this.capacity];
    }

    /**
     * 判断队列是否为空
     *
     * @return
     */
    public boolean isEmpty() {
        return this.curSize == 0;
    }

    /**
     * 判断队列是否已经满了
     *
     * @return
     */
    public boolean isFull() {
        return this.curSize == this.capacity;
    }
}
