import React, { useEffect, useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import { BsArrowLeft, BsArrowRight, BsDot } from 'react-icons/bs';
import { Link } from 'react-router-dom';

const CustomTable = ({ title }) => {
    const [inquiries, setInquiries] = useState([]);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 5;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedInquiries = inquiries.slice(startIndex, endIndex);

    useEffect(() => {
        fetchInquiries();
    }, [refreshKey]);

    const fetchInquiries = async () => {
        try {
            const token = getToken();
            const response = await axios.get(`${BASE_URL}/custom-inquiries`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setInquiries(response.data);
        } catch (err) {
            console.error("Failed to load custom inquiries", err);
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
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Phone</th>
                                    <th>Vehicle</th>
                                    <th>People</th>
                                    <th>Days</th>
                                    <th>From</th>
                                    <th>To</th>
                                    <th>Hotel Rate</th>
                                    <th>Locations</th>
                                    <th>Languages</th>
                                    <th>Remark</th>
                                </tr>
                            </thead>
                            <tbody>
                                {paginatedInquiries.map((inq, idx) => (
                                    <tr key={inq._id}>
                                        <td>{startIndex + idx + 1}</td>
                                        <td>{inq.name}</td>
                                        <td>{inq.email}</td>
                                        <td>{inq.contactNumber}</td>
                                        <td>{inq.vehicleType}</td>
                                        <td>{inq.peopleCount}</td>
                                        <td>{inq.dateCount}</td>
                                        <td>{inq.from ? new Date(inq.from).toLocaleDateString() : ''}</td>
                                        <td>{inq.to ? new Date(inq.to).toLocaleDateString() : ''}</td>
                                        <td>{inq.expectedHotelRate}</td>
                                        <td>
                                            {inq.locationList && inq.locationList.length > 0
                                                ? <ul className="mb-0 ps-3">{inq.locationList.map((loc, i) => (
                                                    <li key={i}>{loc}</li>
                                                ))}</ul>
                                                : <span className="text-muted">-</span>
                                            }
                                        </td>
                                        <td>
                                            {inq.preferredLanguages && inq.preferredLanguages.length > 0
                                                ? <ul className="mb-0 ps-3">{inq.preferredLanguages.map((lang, i) => (
                                                    <li key={i}>{lang}</li>
                                                ))}</ul>
                                                : <span className="text-muted">-</span>
                                            }
                                        </td>
                                        <td>{inq.remark || <span className="text-muted">-</span>}</td>
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
                        {Array.from({ length: Math.ceil(inquiries.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(inquiries.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(inquiries.length / itemsPerPage) - 1)) {
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
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, Math.ceil(inquiries.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(inquiries.length / itemsPerPage) ? 'disabled' : ''}
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

export default CustomTable;
