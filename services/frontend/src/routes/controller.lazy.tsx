import { createLazyFileRoute } from '@tanstack/react-router'
import GameController from '../pages/controller-page'

export const Route = createLazyFileRoute('/controller')({
  component: () => <GameController />,
})