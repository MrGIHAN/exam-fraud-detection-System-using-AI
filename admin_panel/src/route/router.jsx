import { createBrowserRouter } from "react-router-dom";
import RootLayout from "../layout/root";
import Home from "../pages/home";
import LayoutAuth from "../layout/layoutAuth";
import LoginCreative from "../pages/login-creative";
import ResetCreative from "../pages/reset-creative";

//--------------------------------------------------
import PrivateRoute from "../utils/PrivateRoute";
import Subscribe from "../pages/subscribe/subscribe";
import UserDetails from "../pages/user/user-details";

import AddLocation from "../pages/location/add-location";
import Locations from "../pages/location/locations";
import UpdateLocation from "../pages/location/update-location";


import Booking from "../pages/booking/booking";
import Custom from "../pages/custom/custom";
import Package from "../pages/package/package";
import AddPackage from "../pages/package/add-package";
import UpdatePackage from "../pages/package/update-package";

export const router = createBrowserRouter([
    {
        path: "/",
        element: (
            <PrivateRoute>
              <RootLayout />
            </PrivateRoute>
          ),
        children: [
            {
                path: "/",
                element: <Home />
            },
            {
                path: "/admin/sub",
                element: <Subscribe />
            },
            {
                path: "/admin/user-list",
                element: <UserDetails />
            },
            {
                path: "/admin/booking",
                element: <Booking />
            },
            {
                path: "/admin/package",
                element: <Package />
            },
            {
                path: "/admin/package/create",
                element: <AddPackage />
            },
            {
                path: "/admin/package/edit/:id",
                element: <UpdatePackage />
            },
            {
                path: "/admin/custom",
                element: <Custom />
            },
            {
                path: "/admin/location",
                element: <Locations />
            },
            {
                path: "/admin/location/create",
                element: <AddLocation />
            },
            {
                path: "/admin/location/edit/:id",
                element: <UpdateLocation />
            },
        ]
    },
    {
        path: "/",
        element: <LayoutAuth />,
        children: [
            {
                path: "/authentication/login/creative",
                element: <LoginCreative />
            },
            {
                path: "/authentication/reset/creative",
                element: <ResetCreative />
            },
        ]
    }
])