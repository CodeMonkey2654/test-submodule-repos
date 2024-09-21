import { createLazyFileRoute } from '@tanstack/react-router'
import HomePage from '../pages/main-dashboard'

export const Route = createLazyFileRoute('/')({
  component: () => <HomePage />,
})

