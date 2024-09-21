import { createLazyFileRoute } from '@tanstack/react-router'
import AnalyticsPage from '../pages/analytics-page'

export const Route = createLazyFileRoute('/analytics')({
  component: () => <AnalyticsPage />,
})