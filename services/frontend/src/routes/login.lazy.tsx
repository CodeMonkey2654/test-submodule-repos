import { createLazyFileRoute } from '@tanstack/react-router'
import Login from '../pages/login-page'

export const Route = createLazyFileRoute('/login')({
  component: () => <Login />,
})

