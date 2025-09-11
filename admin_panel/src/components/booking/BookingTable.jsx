import React, { useEffect, useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import { BsArrowLeft, BsArrowRight, BsDot } from 'react-icons/bs';
import Swal from 'sweetalert2';
import { Link } from 'react-router-dom';

const STATUS_OPTIONS = [
    { value: "pending", label: "Pending" },
    { value: "cancel", label: "Cancel" },
    { value: "confirm", label: "Confirm" }
];

const BookingTable = ({ title }) => {
    const [bookings, setBookings] = useState([]);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 5;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedBookings = bookings.slice(startIndex, endIndex);

    useEffect(() => {
        fetchBookings();
    }, [refreshKey]);

    const fetchBookings = async () => {
        try {
            const token = getToken();
            const response = await axios.get(`${BASE_URL}/bookings`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setBookings(response.data);
        } catch (err) {
            console.error("Failed to load bookings", err);
        }
    };

    const handleStatusChange = async (bookingId, newStatus) => {
        try {
            const token = getToken();
            await axios.put(`${BASE_URL}/bookings/${bookingId}`, { status: newStatus }, {
                headers: { Authorization: `Bearer ${token}` }
            });
            Swal.fire('Updated!', 'Booking status updated.', 'success');
            fetchBookings();
        } catch (err) {
            Swal.fire('Error!', 'Failed to update status.', 'error');
        }
    };

    if (isRemoved) return null;

    return (
        <div className="col-xxl-12">
            <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
                <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
                <div className="card-body custom-card-action p-0">
                    <div className="table-responsive">
                        <table className="table table-hover mb-0">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Tour</th>
                                    <th>User</th>
                                    <th>Email</th>
                                    <th>Phone</th>
                                    <th>People</th>
                                    <th>From</th>
                                    <th>To</th>
                                    <th>Days</th>
                                    <th>Total</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {paginatedBookings.map((b, idx) => (
                                    <tr key={b._id}>
                                        <td>{startIndex + idx + 1}</td>
                                        <td>
                                            {b.packageid
                                                ? <>
                                                    <div>{b.packageid.title}</div>
                                                    {b.packageid.imgurl &&
                                                        <img src={b.packageid.imgurl} alt="tour" style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 4 }} />}
                                                </>
                                                : <span className="text-muted">N/A</span>
                                            }
                                        </td>
                                        <td>
                                            {b.userid
                                                ? `${b.userid.firstname} ${b.userid.lastname}`
                                                : <span className="text-muted">Guest</span>
                                            }
                                        </td>
                                        <td>{b.contactemail}</td>
                                        <td>{b.contactnu}</td>
                                        <td>{b.totalpeople}</td>
                                        <td>{b.from ? new Date(b.from).toLocaleDateString() : ''}</td>
                                        <td>{b.to ? new Date(b.to).toLocaleDateString() : ''}</td>
                                        <td>{b.totaldays}</td>
                                        <td>{b.totalprice}</td>
                                        <td>
                                            <select
                                                className="form-select"
                                                value={b.status}
                                                onChange={e => handleStatusChange(b._id, e.target.value)}
                                            >
                                                {STATUS_OPTIONS.map(opt => (
                                                    <option key={opt.value} value={opt.value}>
                                                        {opt.label}
                                                    </option>
                                                ))}
                                            </select>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
                {/* Pagination */}
                <div className="card-footer">
                    <ul className="list-unstyled d-flex align-items-center gap-2 mb-0 pagination-common-style">
                        <li>
                            <Link
                                to="#"
                                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                                className={currentPage === 1 ? 'disabled' : ''}
                            >
                                <BsArrowLeft size={16} />
                            </Link>
                        </li>
                        {Array.from({ length: Math.ceil(bookings.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(bookings.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(bookings.length / itemsPerPage) - 1)) {
                                return (
                                    <li key={`dots-${index}`}>
                                        <Link to="#" onClick={e => e.preventDefault()}>
                                            <BsDot size={16} />
                                        </Link>
                                    </li>
                                );
                            }
                            return shouldShow ? (
                                <li key={index}>
                                    <Link
                                        to="#"
                                        onClick={() => setCurrentPage(page)}
                                        className={currentPage === page ? 'active' : ''}
                                    >
                                        {page}
                                    </Link>
                                </li>
                            ) : null;
                        })}
                        <li>
                            <Link
                                to="#"
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, Math.ceil(bookings.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(bookings.length / itemsPerPage) ? 'disabled' : ''}
                            >
                                <BsArrowRight size={16} />
                            </Link>
                        </li>
                    </ul>
                </div>
                <CardLoader refreshKey={refreshKey} />
            </div>
        </div>
    );
};

export default BookingTable;
