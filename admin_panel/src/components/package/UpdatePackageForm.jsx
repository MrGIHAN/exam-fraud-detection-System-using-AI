import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import Swal from 'sweetalert2';

const UpdatePackageForm = ({ title }) => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        title: '',
        popular: 0,
        overview: '',
        days: '',
        people: '',
        activityType: '',
        price: '',
    });
    const [highlights, setHighlights] = useState(['']);
    const [include, setInclude] = useState(['']);
    const [imgFile, setImgFile] = useState(null); // for new upload
    const [imgPreview, setImgPreview] = useState(''); // existing URL
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

    // Fetch package data
    useEffect(() => {
        fetchPackage();
        // eslint-disable-next-line
    }, [id]);

    const fetchPackage = async () => {
        try {
            const token = getToken();
            const res = await axios.get(`${BASE_URL}/packages/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const pkg = res.data;
            setFormData({
                title: pkg.title || '',
                popular: pkg.popular || 0,
                overview: pkg.overview || '',
                days: pkg.days || '',
                people: pkg.people || '',
                activityType: pkg.activityType || '',
                price: pkg.price || '',
            });
            setHighlights(pkg.highlights && Array.isArray(pkg.highlights) && pkg.highlights.length ? pkg.highlights : ['']);
            setInclude(pkg.include && Array.isArray(pkg.include) && pkg.include.length ? pkg.include : ['']);
            setImgPreview(pkg.imgurl || '');
        } catch (err) {
            console.error("Error fetching package", err);
            Swal.fire("Error", "Could not fetch package data", "error");
        }
    };

    // Dynamic arrays
    const handleDynamicChange = (arr, setArr, idx, value) => {
        setArr(prev => prev.map((item, i) => (i === idx ? value : item)));
    };
    const handleAddParagraph = (arr, setArr) => setArr(prev => [...prev, '']);
    const handleRemoveParagraph = (arr, setArr, idx) => setArr(prev => prev.filter((_, i) => i !== idx));

    // Form fields
    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };
    const handlePopularChange = (e) => {
        setFormData(prev => ({ ...prev, popular: e.target.checked ? 1 : 0 }));
    };
    const handleImageChange = (e) => {
        setImgFile(e.target.files[0]);
        setImgPreview(URL.createObjectURL(e.target.files[0]));
    };

    // Submit
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            const token = getToken();
            const data = new FormData();
            data.append('title', formData.title);
            data.append('popular', formData.popular);
            data.append('overview', formData.overview);
            data.append('days', formData.days);
            data.append('people', formData.people);
            data.append('activityType', formData.activityType);
            data.append('price', formData.price);
            data.append('highlights', highlights.map(x => x.trim()).join('|'));
            data.append('include', include.map(x => x.trim()).join('|'));
            // Only send if changed
            if (imgFile) data.append('imgurl', imgFile);

            await axios.put(`${BASE_URL}/packages/${id}`, data, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data',
                },
            });

            Swal.fire({
                icon: 'success',
                title: 'Updated!',
                text: 'Package updated successfully!',
                timer: 1500,
                showConfirmButton: false
            });

            setTimeout(() => {
                navigate('/admin/package');
            }, 1600);

        } catch (err) {
            console.error("Update failed", err);
            Swal.fire("Error", "Failed to update package", "error");
        } finally {
            setIsSubmitting(false);
        }
    };

    if (isRemoved) return null;

    return (
        <div className="col-xxl-12">
            <div className={`card stretch stretch-full ${isExpanded ? "card-expand" : ""} ${refreshKey ? "card-loading" : ""}`}>
                <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />
                <form onSubmit={handleSubmit} encType="multipart/form-data">
                    <div className="card-body">
                        <div className="mb-3">
                            <label className="form-label">Title</label>
                            <input type="text" className="form-control" name="title" value={formData.title} onChange={handleChange} required />
                        </div>
                        <div className="mb-3 form-check">
                            <input className="form-check-input" type="checkbox" checked={formData.popular === 1} onChange={handlePopularChange} id="popularCheck" />
                            <label className="form-check-label" htmlFor="popularCheck">Popular</label>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Overview</label>
                            <textarea className="form-control" name="overview" value={formData.overview} onChange={handleChange} required rows={2} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Highlights</label>
                            {highlights.map((val, idx) => (
                                <div key={idx} className="d-flex align-items-center mb-2">
                                    <textarea
                                        className="form-control" rows="1"
                                        value={val}
                                        onChange={e => handleDynamicChange(highlights, setHighlights, idx, e.target.value)}
                                        required
                                    />
                                    {highlights.length > 1 &&
                                        <button type="button" className="btn btn-danger ms-2" onClick={() => handleRemoveParagraph(highlights, setHighlights, idx)}>
                                            Remove
                                        </button>}
                                </div>
                            ))}
                            <button type="button" className="btn btn-secondary" onClick={() => handleAddParagraph(highlights, setHighlights)}>
                                Add Highlight
                            </button>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Include</label>
                            {include.map((val, idx) => (
                                <div key={idx} className="d-flex align-items-center mb-2">
                                    <textarea
                                        className="form-control" rows="1"
                                        value={val}
                                        onChange={e => handleDynamicChange(include, setInclude, idx, e.target.value)}
                                        required
                                    />
                                    {include.length > 1 &&
                                        <button type="button" className="btn btn-danger ms-2" onClick={() => handleRemoveParagraph(include, setInclude, idx)}>
                                            Remove
                                        </button>}
                                </div>
                            ))}
                            <button type="button" className="btn btn-secondary" onClick={() => handleAddParagraph(include, setInclude)}>
                                Add Include
                            </button>
                        </div>
                        <div className="mb-3 row">
                            <div className="col-md-4">
                                <label className="form-label">Days</label>
                                <input type="number" className="form-control" name="days" value={formData.days} onChange={handleChange} min="1" required />
                            </div>
                            <div className="col-md-4">
                                <label className="form-label">People</label>
                                <input type="number" className="form-control" name="people" value={formData.people} onChange={handleChange} min="1" required />
                            </div>
                            <div className="col-md-4">
                                <label className="form-label">Price</label>
                                <input type="number" className="form-control" name="price" value={formData.price} onChange={handleChange} min="0" required />
                            </div>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Activity Type</label>
                            <input type="text" className="form-control" name="activityType" value={formData.activityType} onChange={handleChange} />
                            <small className="text-muted">For multiple, separate by | (e.g., "Cultural|Adventure")</small>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Image</label>
                            <input type="file" className="form-control" accept="image/*" onChange={handleImageChange} />
                            {imgPreview &&
                                <div className="mt-2">
                                    <img src={imgPreview} alt="Preview" style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #ccc' }} />
                                </div>
                            }
                        </div>
                    </div>
                    <div className="card-footer d-flex justify-content-end">
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? 'Saving...' : 'Update Package'}
                        </button>
                    </div>
                </form>
                <CardLoader refreshKey={refreshKey} />
            </div>
        </div>
    );
};

export default UpdatePackageForm;
