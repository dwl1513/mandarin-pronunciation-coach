import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 开发服务器收到退出信号时，强制断开所有长连接（主要是浏览器的 HMR WebSocket）。
// Node 的 server.close() 默认会等已有连接自己结束，而 HMR WebSocket 永不主动断开，
// 导致 Ctrl+C 后要卡几十秒才退出。这里一刀切断，立即退出。仅 dev 生效，不影响构建。
function fastShutdown(): Plugin {
  return {
    name: 'fast-shutdown',
    configureServer(server) {
      const close = () => {
        // dev 下 httpServer 是 Node 原生 http.Server，类型联合里多了 Http2 分支才需断言。
        const httpServer = server.httpServer as { closeAllConnections?: () => void } | null
        httpServer?.closeAllConnections?.()
        server.close().finally(() => process.exit(0))
      }
      process.once('SIGINT', close)
      process.once('SIGTERM', close)
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), fastShutdown()],
})
