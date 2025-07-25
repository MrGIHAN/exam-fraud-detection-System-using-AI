import React, { useEffect, useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import { BsArrowLeft, BsArrowRight, BsDot } from 'react-icons/bs';
import { Link } from 'react-router-dom';
import Swal from 'sweetalert2';
import { Button } from 'react-bootstrap';
import { FiCheckCircle } from 'react-icons/fi';

const UserDetailTable = ({ title }) => {
    const [users, setUsers] = useState([]);
    const [editingRole, setEditingRole] = useState({});
    const [savingRole, setSavingRole] = useState({});
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedUsers = users.slice(startIndex, endIndex);

    useEffect(() => {
        fetchUsers();
    }, [refreshKey]);

    const fetchUsers = async () => {
        try {
            const userRes = await axios.get(`${BASE_URL}/student`);

            setUsers(userRes.data);
        } catch (err) {
            console.error("Failed to load users", err);
        }
    };

    const handleRoleChange = (userId, newUsertype) => {
        setEditingRole(prev => ({ ...prev, [userId]: Number(newUsertype) }));
    };

    /*const handleRoleSave = async (user) => {
        const newUsertype = editingRole[user._id];
        if (!newUsertype || newUsertype === user.usertype) return;
        setSavingRole(prev => ({ ...prev, [user._id]: true }));

        try {
            const token = getToken();
            // Only updating usertype, send the full user object if needed by backend
            const updatedUser = {
                firstname: user.firstname,
                lastname: user.lastname,
                email: user.email,
                contactNumber: user.contactNumber,
                usertype: newUsertype,
            };
            await axios.put(
                `${BASE_URL}/users/${user._id}`,
                updatedUser,
                { headers: { Authorization: `Bearer ${token}` } }
            );
            Swal.fire({ icon: 'success', title: 'Role updated', timer: 1200, showConfirmButton: false });
            setEditingRole(prev => ({ ...prev, [user._id]: undefined }));
            fetchUsers();
        } catch (err) {
            Swal.fire('Error!', 'Failed to update role.', 'error');
        } finally {
            setSavingRole(prev => ({ ...prev, [user._id]: false }));
        }
    };*/

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
                                    <th style={{ display: 'none' }}>ID</th>
                                    <th>ID</th>
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Phone</th>
                                    
                                    <th>Number of sessions attended</th>
                                </tr>
                            </thead>
                            <tbody>
                                {paginatedUsers.map((user) => (
                                    <tr key={user._id}>
                                        <td style={{ display: 'none' }}>{user._id}</td>
                                        <td>{user.studentId}</td>
                                        <td>{user.nameF} {user.nameL}</td>
                                        <td>{user.email}</td>
                                        <td>{user.createdAt}</td>
                                        <td>{user.sessionCount}</td>
                                        {/* Role dropdown + Save */}
                                        
                                        
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
                        {Array.from({ length: Math.ceil(users.length / itemsPerPage) }, (_, index) => {
                            const page = index + 1;
                            const shouldShow =
                                page === 1 ||
                                page === Math.ceil(users.length / itemsPerPage) ||
                                Math.abs(currentPage - page) <= 1;
                            if (!shouldShow && (page === 2 || page === Math.ceil(users.length / itemsPerPage) - 1)) {
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
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, Math.ceil(users.length / itemsPerPage)))}
                                className={currentPage === Math.ceil(users.length / itemsPerPage) ? 'disabled' : ''}
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

export default UserDetailTable;
