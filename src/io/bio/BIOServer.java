package io.bio;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * 同步阻塞通信
 */
public class BIOServer {

    public static void main(String[] args) {
        try {
            System.out.println("服务端启动！");
            ServerSocket server = new ServerSocket(9999);
            Socket socket = server.accept();
            BufferedReader bis = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            String msg = "";
            if ((msg = bis.readLine()) != null) {
                System.out.println("服务端收到的消息是：" + msg);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
