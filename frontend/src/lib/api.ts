// 后端地址。开发期默认指向本地 FastAPI；可用 VITE_API_BASE 覆盖。
// 去掉了 vite proxy，前端直接请求后端绝对地址（后端已配置 CORS 放行 5173）。
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"

/** 拼接后端 API 路径，例如 api("/api/health") */
export function api(path: string): string {
  return `${API_BASE}${path}`
}

/** 把后端返回的相对资源路径（图片 / 音频）补成绝对地址；空值原样返回 */
export function resolveAsset(url: string | null | undefined): string {
  if (!url) {
    return ""
  }
  if (/^https?:\/\//.test(url)) {
    return url
  }
  return url.startsWith("/api") ? `${API_BASE}${url}` : url
}
