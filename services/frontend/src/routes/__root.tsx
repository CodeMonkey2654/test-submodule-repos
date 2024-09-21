import { createRootRoute, Link, Outlet } from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/router-devtools'

export const Route = createRootRoute({
  component: () => (
    <>
      <nav className="bg-blue-600 p-4 shadow-md">
        <div className="container mx-auto flex justify-between items-center">
          <div className="text-black font-bold text-xl">RoverView</div>
          <div className="flex space-x-4">
            <Link to="/" className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Home
            </Link>
            <Link to='/login' className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Login
            </Link>
            <Link to='/signup' className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Sign Up
            </Link>
            <Link to="/controller" className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Controller
            </Link>
            <Link to="/analytics" className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Analysis
            </Link>
            <Link to="/help" className="text-black hover:text-blue-200 transition duration-300 [&.active]:font-bold">
              Help
            </Link>
          </div>
        </div>
      </nav>
      <Outlet />
      <TanStackRouterDevtools />
    </>
  ),
})
