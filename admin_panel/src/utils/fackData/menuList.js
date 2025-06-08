export const menuList = [
    {
        id: 0,
        name: "dashboards",
        path: "/",
        icon: 'feather-monitor',
        dropdownMenu: [
            {
                id: 1,
                name: "CRM",
                path: "/",
                subdropdownMenu: false
            }
        ]
    },
    {
        id: 1,
        name: "Exam Sessions",
        path: "#",
        icon: 'feather-layers',
        dropdownMenu: [
            {
                id: 1,
                name: "Session List",
                path: "/admin/sub",
                subdropdownMenu: false
            },

        ]
    },
    {
        id: 2,
        name: "Students",
        path: "#",
        icon: 'feather-users',
        dropdownMenu: [
            {
                id: 1,
                name: "Student List",
                path: "/admin/user-list",
                subdropdownMenu: false
            },

        ]
    },
    // {
    //     id: 4,
    //     name: "Custom inquires",
    //     path: "#",
    //     icon: 'feather-award',
    //     dropdownMenu: [
    //         {
    //             id: 1,
    //             name: "Custom inquires List",
    //             path: "/admin/custom",
    //             subdropdownMenu: false
    //         },

    //     ]
    // },
    // {
    //     id: 5,
    //     name: "Bookings",
    //     path: "#",
    //     icon: 'feather-layers',
    //     dropdownMenu: [
    //         {
    //             id: 1,
    //             name: "Booking List",
    //             path: "/admin/booking",
    //             subdropdownMenu: false
    //         },


    //     ]
    // },
    // {
    //     id: 6,
    //     name: "Location List",
    //     path: "#",
    //     icon: 'feather-archive',
    //     dropdownMenu: [
    //         {
    //             id: 1,
    //             name: "Location List",
    //             path: "/admin/location",
    //             subdropdownMenu: false
    //         },
    //         {
    //             id: 2,
    //             name: "Add Location",
    //             path: "/admin/location/create",
    //             subdropdownMenu: false
    //         },

    //     ]
    // },
    // {
    //     id: 7,
    //     name: "Package List",
    //     path: "#",
    //     icon: 'feather-briefcase',
    //     dropdownMenu: [
    //         {
    //             id: 1,
    //             name: "Package List",
    //             path: "/admin/package",
    //             subdropdownMenu: false
    //         },
    //         {
    //             id: 2,
    //             name: "Add New Package",
    //             path: "/admin/package/create",
    //             subdropdownMenu: false
    //         }
            

    //     ]
    // },
    
]
