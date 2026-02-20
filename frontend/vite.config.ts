import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Use process.cwd() to find root or go up from current dir
  const env = loadEnv(mode, path.resolve(process.cwd(), '..'), '')
  const port = env.PORT || '8000'
  const target = `http://localhost:${port}`

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/token': target,
        '/upload': target,
        '/mcp': {
          target: target,
          ws: true
        }
      }
    }
  }
})
