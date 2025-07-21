import React, { useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import Swal from 'sweetalert2';

const AddPackageForm = ({ title }) => {
    const [formData, setFormData] = useState({
        nameF: '',
        nameL: '',
        email: '',
        password: '',
    });
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            const response = await axios.post(`${BASE_URL}/student/register`, formData);

            Swal.fire({
                icon: 'success',
                title: 'Student Registered',
                text: response.data.message,
                timer: 2000,
                showConfirmButton: false,
            });

            setFormData({ nameF: '', nameL: '', email: '', password: '' });
            handleRefresh();
        } catch (err) {
            Swal.fire({
                icon: 'error',
                title: 'Registration Failed',
                text: err.response?.data?.error || 'An error occurred',
            });
        } finally {
            setIsSubmitting(false);
        }
    };

    if (isRemoved) return null;

    return (
        <div className="col-xxl-12">
            <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
                <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
                <form onSubmit={handleSubmit}>
                    <div className="card-body">
                        <div className="mb-3">
                            <label className="form-label">First Name</label>
                            <input type="text" className="form-control" name="nameF" value={formData.nameF} onChange={handleChange} required />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Last Name</label>
                            <input type="text" className="form-control" name="nameL" value={formData.nameL} onChange={handleChange} required />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Email</label>
                            <input type="email" className="form-control" name="email" value={formData.email} onChange={handleChange} required />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Password</label>
                            <input type="password" className="form-control" name="password" value={formData.password} onChange={handleChange} required />
                        </div>
                    </div>
                    <div className="card-footer d-flex justify-content-end">
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? 'Registering...' : 'Register Student'}
                        </button>
                    </div>
                </form>
                <CardLoader refreshKey={refreshKey} />
            </div>
        </div>
    );
};


export default AddPackageForm;
