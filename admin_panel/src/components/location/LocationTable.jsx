import React, { useEffect, useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import Dropdown from '@/components/shared/Dropdown';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import { BsArrowLeft, BsArrowRight, BsDot } from 'react-icons/bs';
import { FiEdit3, FiTrash2 } from 'react-icons/fi';
import { Link, useNavigate } from 'react-router-dom';
import Swal from 'sweetalert2';

const LocationTable = ({ title }) => {
    const navigate = useNavigate();
    const [locations, setLocations] = useState([]);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedLocations = locations.slice(startIndex, endIndex);

    useEffect(() => {
        fetchLocations();
    }, [refreshKey]);

    const fetchLocations = async () => {
        try {
            const token = getToken();
            const res = await axios.get(`${BASE_URL}/locations`, { headers: { Authorization: `Bearer ${token}` } });
            setLocations(res.data);
        } catch (err) {
            console.error("Failed to load locations", err);
        }
    };

    const handleDeleteLocation = async (locationId) => {
        const confirm = await Swal.fire({
            title: 'Are you sure?',
            text: "You won't be able to revert this!",
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#6c757d',
            confirmButtonText: 'Yes, delete it!'
        });

        if (confirm.isConfirmed) {
            try {
                const token = getToken();
                await axios.delete(`${BASE_URL}/locations/${locationId}`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                Swal.fire('Deleted!', 'Location has been deleted.', 'success');
                fetchLocations();
            } catch (err) {
                Swal.fire('Error!', 'Failed to delete location.', 'error');
            }
        }
    };

    const handleActionClick = (action, locationId) => {
        if (action === 'Delete') {
            handleDeleteLocation(locationId);
        } else if (action === 'Edit') {
            navigate(`/admin/location/edit/${locationId}`);
        }
    };

    const getDropdownItems = (locationId) => [
        {
            icon: <FiEdit3 />,
            label: 'Edit',
            onClick: () => handleActionClick('Edit', locationId)
        },
        { type: "divider" },
        {
            icon: <FiTrash2 />,
            label: 'Delete',
            onClick: () => handleActionClick('Delete', locationId)
        }
    ];

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
                                    <th>Main Image</th>
                                    <th>Title</th>
                                    <th>Description</th>
                                    <th>Map</th>
                                    <th>Other Images</th>
                                    <th className="text-end">Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {paginatedLocations.map(location => (
                                    <tr key={location._id}>
                                        <td>
                                            <img
                                                src={location.mainimage}
                                                alt={location.title}
                                                style={{
                                                    width: '60px',
                                                    height: '60px',
                                                    objectFit: 'cover',
                                                    borderRadius: '8px'
                                                }}
                                            />
                                        </td>
                                        <td style={{ minWidth: '150px', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                                            title={location.title}
                                        >
                                            {location.title}
                                        </td>
                                        <td style={{ minWidth: '200px', maxWidth: '350px', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}
                                            title={location.descriptions && location.descriptions[0]}
                                        >
                                            {location.descriptions && location.descriptions[0]
                                                ? location.descriptions[0].length > 100
                                                    ? `${location.descriptions[0].substring(0, 100)}...`
                                                    : location.descriptions[0]
                                                : ''}
                                        </td>
                                        <td>
                                            <a href={location.mapurl} target="_blank" rel="noopener noreferrer">
                                                Google Maps
                                            </a>
                                        </td>
                                        <td>
                                            <div className="d-flex gap-1 flex-wrap">
                                                {(location.otherimages || []).map((img, i) => (
                                                    <img key={i} src={img} alt="other" style={{
                                                        width: '40px',
                                                        height: '40px',
                                                        objectFit: 'cover',
                                                        borderRadius: '5px'
                                                    }} />
                                                ))}
                                            </div>
                                        </td>
                                        <td className="text-end">
                                            <Dropdown
                                                dropdownItems={getDropdownItems(location._id)}
                                                triggerClass="avatar-md ms-auto"
                                                triggerPosition="0,28"
                                            />
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

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
                        {Array.from({ length: Math.ceil(locations.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(locations.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(locations.length / itemsPerPage) - 1)) {
                                return (
                                    <li key={`dots-${index}`}>
                                        <Link to="#" onClick={(e) => e.preventDefault()}>
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
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, Math.ceil(locations.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(locations.length / itemsPerPage) ? 'disabled' : ''}
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

export default LocationTable;
