import { createLazyFileRoute } from '@tanstack/react-router'
import HelpPage from '../pages/help-page'

export const Route = createLazyFileRoute('/help')({
  component: () => <HelpPage />,
})
