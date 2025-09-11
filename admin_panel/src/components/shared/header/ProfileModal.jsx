import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FiUser, FiSettings, FiLogOut } from "react-icons/fi";
import { clearToken, getUserName } from '@/utils/token';

const ProfileModal = () => {
    const navigate = useNavigate();
    const userName = getUserName() || "User";

    const handleLogout = () => {
        clearToken(); // Clear all token info
        navigate('/authentication/login/creative'); // Redirect
    };

    return (
        <div className="dropdown nxl-h-item">
            <a href="#" data-bs-toggle="dropdown" role="button" data-bs-auto-close="outside">
                <img src="/images/avatar/1.png" alt="user-image" className="img-fluid user-avtar me-0" />
            </a>
            <div className="dropdown-menu dropdown-menu-end nxl-h-dropdown nxl-user-dropdown">
                <div className="dropdown-header">
                    <div className="d-flex align-items-center">
                        <img src="/images/avatar/1.png" alt="user-image" className="img-fluid user-avtar" />
                        <div>
                            <h6 className="text-dark mb-0">{userName}<span className="badge bg-soft-success text-success ms-1">PRO</span></h6>
                            {/* <span className="fs-12 fw-medium text-muted">alex.della@outlook.com</span> */}
                        </div>
                    </div>
                </div>

                <a href="#" className="dropdown-item">
                    <i><FiUser /></i>
                    <span>Profile Details</span>
                </a>

                {/* <a href="#" className="dropdown-item">
                    <i><FiSettings /></i>
                    <span>Account Settings</span>
                </a> */}

                <div className="dropdown-divider"></div>

                <button onClick={handleLogout} className="dropdown-item" style={{ background: 'none', border: 'none', width: '100%', textAlign: 'start' }}>
                    <i><FiLogOut /></i>
                    <span>Logout</span>
                </button>
            </div>
        </div>
    );
};

export default ProfileModal;
