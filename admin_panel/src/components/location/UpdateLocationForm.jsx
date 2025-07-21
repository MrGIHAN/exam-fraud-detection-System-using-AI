import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import CardHeader from '@/components/shared/CardHeader';
import CardLoader from '@/components/shared/CardLoader';
import useCardTitleActions from '@/hooks/useCardTitleActions';
import axios from 'axios';
import BASE_URL from '../../config/apiConfig';
import { getToken } from '@/utils/token';
import Swal from 'sweetalert2';

const UpdateLocationForm = ({ title }) => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        title: '',
        mapurl: '',
    });
    const [descriptions, setDescriptions] = useState(['']);
    const [mainImage, setMainImage] = useState(null); // New upload
    const [mainImagePreview, setMainImagePreview] = useState(''); // Existing main image URL
    const [otherImages, setOtherImages] = useState([]); // New uploads
    const [existingOtherImages, setExistingOtherImages] = useState([]); // Existing URLs
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { refreshKey, isRemoved, isExpanded, handleRefresh, handleExpand, handleDelete } = useCardTitleActions();

    // Fetch data
    useEffect(() => {
        fetchLocation();
    }, [id]);

    const fetchLocation = async () => {
        try {
            const token = getToken();
            const res = await axios.get(`${BASE_URL}/locations/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            const loc = res.data;
            setFormData({
                title: loc.title || '',
                mapurl: loc.mapurl || '',
            });
            setDescriptions(loc.descriptions && Array.isArray(loc.descriptions) ? loc.descriptions : []);
            setMainImagePreview(loc.mainimage || '');
            setExistingOtherImages(loc.otherimages || []);
        } catch (err) {
            console.error("Error fetching location", err);
            Swal.fire("Error", "Could not fetch location data", "error");
        }
    };

    // Handlers
    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleDescriptionChange = (idx, value) => {
        setDescriptions(prev => prev.map((desc, i) => i === idx ? value : desc));
    };
    const handleAddParagraph = () => setDescriptions(prev => [...prev, '']);
    const handleRemoveParagraph = (idx) => setDescriptions(prev => prev.filter((_, i) => i !== idx));

    const handleMainImageChange = (e) => {
        setMainImage(e.target.files[0]);
        setMainImagePreview(URL.createObjectURL(e.target.files[0]));
    };

    const handleOtherImagesChange = (e) => {
        setOtherImages(Array.from(e.target.files));
    };

    const handleRemoveExistingOtherImage = (idx) => {
        setExistingOtherImages(prev => prev.filter((_, i) => i !== idx));
    };

    // Submit
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            const token = getToken();
            const data = new FormData();
            data.append('title', formData.title);
            data.append('mapurl', formData.mapurl);
            data.append('descriptions', descriptions.map(d => d.trim()).join('|'));
            // Only upload mainimage if changed
            if (mainImage) data.append('mainimage', mainImage);
            // Only upload new other images if present
            otherImages.forEach(file => data.append('otherimages', file));
            // Existing images sent as URLs, comma-separated (backend should handle)
            data.append('existingOtherImages', existingOtherImages.join(','));
            // If main image preview is still a URL (unchanged), send it
            if (mainImagePreview && !mainImage) data.append('existingMainImage', mainImagePreview);

            await axios.put(`${BASE_URL}/locations/${id}`, data, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data',
                },
            });

            Swal.fire({
                icon: 'success',
                title: 'Updated!',
                text: 'Location updated successfully!',
                timer: 1500,
                showConfirmButton: false
            });

            setTimeout(() => {
                navigate('/admin/location');
            }, 1600);

        } catch (err) {
            console.error("Update failed", err);
            Swal.fire("Error", "Failed to update location", "error");
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
                        {/* Main image */}
                        <div className="mb-3">
                            <label className="form-label">Main Image</label>
                            <input type="file" className="form-control" accept="image/*" onChange={handleMainImageChange} />
                            {mainImagePreview && (
                                <div className="mt-2">
                                    <img src={mainImagePreview} alt="Main Preview" style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '5px' }} />
                                </div>
                            )}
                        </div>
                        {/* Other images */}
                        <div className="mb-3">
                            <label className="form-label">Other Images</label>
                            <input type="file" className="form-control" accept="image/*" multiple onChange={handleOtherImagesChange} />
                            <small className="text-muted">You can upload multiple images.</small>
                            {/* Show existing other images with remove */}
                            <div className="mt-2 d-flex gap-2 flex-wrap">
                                {existingOtherImages.map((img, idx) => (
                                    <div key={idx} style={{ position: 'relative' }}>
                                        <img
                                            src={img}
                                            alt={`Other Existing ${idx}`}
                                            style={{ width: '70px', height: '70px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #ccc' }}
                                        />
                                        <button
                                            type="button"
                                            onClick={() => handleRemoveExistingOtherImage(idx)}
                                            style={{
                                                position: 'absolute',
                                                top: '-5px',
                                                right: '-5px',
                                                background: 'red',
                                                color: 'white',
                                                border: 'none',
                                                borderRadius: '50%',
                                                width: '20px',
                                                height: '20px',
                                                cursor: 'pointer',
                                                fontSize: '12px',
                                                lineHeight: '18px',
                                                padding: 0
                                            }}
                                        >
                                            ×
                                        </button>
                                    </div>
                                ))}
                                {otherImages.map((img, idx) => (
                                    <img
                                        key={`new-${idx}`}
                                        src={URL.createObjectURL(img)}
                                        alt={`Other New ${idx}`}
                                        style={{ width: '70px', height: '70px', objectFit: 'cover', borderRadius: '5px', border: '1px solid #ccc' }}
                                    />
                                ))}
                            </div>
                        </div>
                        {/* Map URL */}
                        <div className="mb-3">
                            <label className="form-label">Google Map URL</label>
                            <input type="text" className="form-control" name="mapurl" value={formData.mapurl} onChange={handleChange} required />
                        </div>
                    </div>
                    <div className="card-footer d-flex justify-content-end">
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? 'Saving...' : 'Update Location'}
                        </button>
                    </div>
                </form>
                <CardLoader refreshKey={refreshKey} />
            </div>
        </div>
    );
};

export default UpdateLocationForm;
