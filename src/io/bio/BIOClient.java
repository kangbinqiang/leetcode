package io.bio;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.Socket;

public class BIOClient {
    public static void main(String[] args) {
        try {
            Socket socket = new Socket("127.0.0.1",9999);
            OutputStream os = socket.getOutputStream();
            PrintStream ps  = new PrintStream(os);
            ps.println("我是客户端！");
            ps.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
