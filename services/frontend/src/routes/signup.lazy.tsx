import { createLazyFileRoute } from '@tanstack/react-router'
import SignupPage from '../pages/signup-page'

export const Route = createLazyFileRoute('/signup')({
  component: () => <SignupPage />,
})