import React, { useEffect, useState } from 'react'
import axios from 'axios'
import BASE_URL from '../../config/apiConfig'
import { getToken } from '@/utils/token'
import { FiInbox, FiBookOpen } from 'react-icons/fi'

const SiteOverviewStatistics = () => {
    const [stats, setStats] = useState({
        totalInquiries: 0,
        totalBookings: 0,
    });

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const token = getToken();
                // Fetch inquiries
                const inqRes = await axios.get(`${BASE_URL}/custom-inquiries`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                // Fetch bookings
                const bookingsRes = await axios.get(`${BASE_URL}/bookings`, {
                    headers: { Authorization: `Bearer ${token}` }
                });

                setStats({
                    totalInquiries: Array.isArray(inqRes.data) ? inqRes.data.length : 0,
                    totalBookings: Array.isArray(bookingsRes.data) ? bookingsRes.data.length : 0,
                });
            } catch (e) {
                setStats({ totalInquiries: 0, totalBookings: 0 });
            }
        };
        fetchStats();
    }, []);

    const data = [
        {
            id: 1,
            icon: <FiInbox />,
            title: 'Total Sessions Today',
            count: 7,
            //label: `${stats.totalInquiries} Inquiry${stats.totalInquiries === 1 ? '' : 'ies'}`,
        },
        {
            id: 2,
            icon: <FiBookOpen />,
            title: 'Total offence detected',
            count: 26,
            //label: `${stats.totalBookings} Booking${stats.totalBookings === 1 ? '' : 's'}`,
        }
    ];

    return (
        <>
            {data.map(({ id, icon, title, count, label }) => (
                <div key={id} className="col-xxl-3 col-md-6">
                    <div className="card stretch stretch-full short-info-card">
                        <div className="card-body">
                            <div className="d-flex align-items-center gap-3 mb-2">
                                <div className="avatar-text avatar-lg bg-gray-200 icon">
                                    {React.cloneElement(icon, { size: 24 })}
                                </div>
                                <div>
                                    <div className="fs-2 fw-bold text-dark">
                                        {count}
                                    </div>
                                    <div className="fs-14 fw-semibold text-truncate-1-line">{title}</div>
                                </div>
                            </div>
                            <div className="text-end pt-2">
                                <span className="fs-12 text-muted">{label}</span>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </>
    );
};

export default SiteOverviewStatistics;
