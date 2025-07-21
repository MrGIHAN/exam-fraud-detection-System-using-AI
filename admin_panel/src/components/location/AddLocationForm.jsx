import React, { useState } from 'react';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import Swal from 'sweetalert2';

const AddLocationForm = ({ title }) => {
    const [formData, setFormData] = useState({
        title: '',
        mapurl: '',
    });

    // For paragraphs
    const [descriptions, setDescriptions] = useState(['']);
    const [mainImage, setMainImage] = useState(null);
    const [otherImages, setOtherImages] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

    // Handle dynamic description fields
    const handleDescriptionChange = (idx, value) => {
        setDescriptions(prev => prev.map((desc, i) => i === idx ? value : desc));
    };
    const handleAddParagraph = () => {
        setDescriptions(prev => [...prev, '']);
    };
    const handleRemoveParagraph = (idx) => {
        setDescriptions(prev => prev.filter((_, i) => i !== idx));
    };

    const handleMainImageChange = (e) => {
        setMainImage(e.target.files[0]);
    };

    const handleOtherImagesChange = (e) => {
        const files = Array.from(e.target.files);
        setOtherImages(files);
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            const token = getToken();
            const payload = new FormData();
            payload.append('title', formData.title);
            payload.append('mapurl', formData.mapurl);

            // <-- Combine all paragraphs into one string, joined by |
            payload.append('descriptions', descriptions.map(d => d.trim()).join('|'));

            if (mainImage) {
                payload.append('mainimage', mainImage);
            }
            otherImages.forEach(file => {
                payload.append('otherimages', file);
            });

            await axios.post(`${BASE_URL}/locations`, payload, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data',
                }
            });

            Swal.fire({
                icon: 'success',
                title: 'Location added!',
                text: 'Location created successfully!',
                timer: 2000,
                showConfirmButton: false,
            });

            setFormData({ title: '', mapurl: '' });
            setDescriptions(['']);
            setMainImage(null);
            setOtherImages([]);
            handleRefresh();
        } catch (err) {
            console.error('Location submission failed:', err?.response?.data || err);
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: err?.response?.data?.error || 'Failed to add location.',
            });
        } finally {
            setIsSubmitting(false);
        }
    };


    if (isRemoved) return null;

    return (
        <div className="col-xxl-12">
            <div className={`card stretch stretch-full ${isExpanded ? 'card-expand' : ''} ${refreshKey ? 'card-loading' : ''}`}>
                <CardHeader title={title} refresh={handleRefresh} remove={handleDelete} expanded={handleExpand} />

                <form onSubmit={handleSubmit} encType="multipart/form-data">
                    <div className="card-body">
                        <div className="mb-3">
                            <label className="form-label">Location Title</label>
                            <input type="text" className="form-control" name="title" value={formData.title} onChange={handleChange} required />
                        </div>

                        {/* Dynamic paragraphs */}
                        <div className="mb-3">
                            <label className="form-label">Descriptions</label>
                            {descriptions.map((desc, idx) => (
                                <div key={idx} className="d-flex align-items-center mb-2">
                                    <textarea
                                        className="form-control"
                                        rows="2"
                                        value={desc}
                                        onChange={e => handleDescriptionChange(idx, e.target.value)}
                                        required
                                    />
                                    {descriptions.length > 1 && (
                                        <button type="button" className="btn btn-danger ms-2" onClick={() => handleRemoveParagraph(idx)}>
                                            Remove
                                        </button>
                                    )}
                                </div>
                            ))}
                            <button type="button" className="btn btn-secondary" onClick={handleAddParagraph}>
                                Add Paragraph
                            </button>
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Main Image</label>
                            <input type="file" className="form-control" accept="image/*" onChange={handleMainImageChange} required />
                            {mainImage && (
                                <div className="mt-2">
                                    <img src={URL.createObjectURL(mainImage)} alt="Main" style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '5px' }} />
                                </div>
                            )}
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Other Images</label>
                            <input type="file" className="form-control" accept="image/*" multiple onChange={handleOtherImagesChange} />
                            <small className="text-muted">You can upload multiple images.</small>
                            <div className="mt-2 d-flex gap-2 flex-wrap">
                                {otherImages.map((img, idx) => (
                                    <img key={idx} src={URL.createObjectURL(img)} alt={`Other ${idx}`} style={{ width: '70px', height: '70px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #ccc' }} />
                                ))}
                            </div>
                        </div>

                        <div className="mb-3">
                            <label className="form-label">Google Map URL</label>
                            <input type="text" className="form-control" name="mapurl" value={formData.mapurl} onChange={handleChange} required />
                        </div>
                    </div>

                    <div className="card-footer d-flex justify-content-end">
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? 'Saving...' : 'Save Location'}
                        </button>
                    </div>
                </form>
                <CardLoader refreshKey={refreshKey} />
            </div>
        </div>
    );
};

export default AddLocationForm;
