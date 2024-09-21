import { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import {
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query'
import UserProvider from './contexts/user-context'
import { routeTree } from './routeTree.gen'
import './index.css'

const router = createRouter({ routeTree })
const queryClient = new QueryClient()

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

// Render the app
const rootElement = document.getElementById('root')!
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement)
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <UserProvider>
          <RouterProvider router={router}/>
        </UserProvider>
      </QueryClientProvider>
    </StrictMode>,
  )
}
