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

const PackageTable = ({ title }) => {
    const navigate = useNavigate();
    const [packages, setPackages] = useState([]);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedPackages = packages.slice(startIndex, endIndex);

    useEffect(() => {
        fetchPackages();
    }, [refreshKey]);

    const fetchPackages = async () => {
        try {
            const token = getToken();
            const res = await axios.get(`${BASE_URL}/packages`, { headers: { Authorization: `Bearer ${token}` } });
            setPackages(res.data);
        } catch (err) {
            console.error("Failed to load packages", err);
        }
    };

    const handleDeletePackage = async (packageId) => {
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
                await axios.delete(`${BASE_URL}/packages/${packageId}`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                Swal.fire('Deleted!', 'Package has been deleted.', 'success');
                fetchPackages();
            } catch (err) {
                Swal.fire('Error!', 'Failed to delete package.', 'error');
            }
        }
    };

    const handleActionClick = (action, packageId) => {
        if (action === 'Delete') {
            handleDeletePackage(packageId);
        } else if (action === 'Edit') {
            navigate(`/admin/package/edit/${packageId}`);
        }
    };

    const getDropdownItems = (packageId) => [
        {
            icon: <FiEdit3 />,
            label: 'Edit',
            onClick: () => handleActionClick('Edit', packageId)
        },
        { type: "divider" },
        {
            icon: <FiTrash2 />,
            label: 'Delete',
            onClick: () => handleActionClick('Delete', packageId)
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
                                    <th>Image</th>
                                    <th>Title</th>
                                    <th>Popular</th>
                                    <th>Overview</th>
                                    <th>Highlights</th>
                                    <th>Include</th>
                                    <th>Activity Type</th>
                                    <th>Price</th>
                                    <th>Days</th>
                                    <th>People</th>
                                    <th className="text-end">Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {paginatedPackages.map(pkg => (
                                    <tr key={pkg._id}>
                                        <td>
                                            {pkg.imgurl &&
                                                <img
                                                    src={pkg.imgurl}
                                                    alt={pkg.title}
                                                    style={{
                                                        width: '60px',
                                                        height: '60px',
                                                        objectFit: 'cover',
                                                        borderRadius: '8px'
                                                    }}
                                                />
                                            }
                                        </td>
                                        <td style={{ minWidth: '120px', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                                            title={pkg.title}
                                        >
                                            {pkg.title}
                                        </td>
                                        <td>
                                            <span className={`badge bg-soft-${pkg.popular === 1 ? 'success' : 'secondary'} text-${pkg.popular === 1 ? 'success' : 'secondary'}`}>
                                                {pkg.popular === 1 ? 'Popular' : 'No'}
                                            </span>
                                        </td>
                                        <td style={{ minWidth: '180px', maxWidth: '300px', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}
                                            title={pkg.overview}
                                        >
                                            {pkg.overview ? (pkg.overview.length > 80 ? pkg.overview.substring(0, 80) + '...' : pkg.overview) : ''}
                                        </td>
                                        <td>
                                            <ul className="mb-0 ps-3">
                                                {pkg.highlights?.map((hl, i) => <li key={i}>{hl}</li>)}
                                            </ul>
                                        </td>
                                        <td>
                                            <ul className="mb-0 ps-3">
                                                {pkg.include?.map((inc, i) => <li key={i}>{inc}</li>)}
                                            </ul>
                                        </td>
                                        <td>{pkg.activityType}</td>
                                        <td>{pkg.price}</td>
                                        <td>{pkg.days}</td>
                                        <td>{pkg.people}</td>
                                        <td className="text-end">
                                            <Dropdown
                                                dropdownItems={getDropdownItems(pkg._id)}
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
                        {Array.from({ length: Math.ceil(packages.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(packages.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(packages.length / itemsPerPage) - 1)) {
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
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, Math.ceil(packages.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(packages.length / itemsPerPage) ? 'disabled' : ''}
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

export default PackageTable;
