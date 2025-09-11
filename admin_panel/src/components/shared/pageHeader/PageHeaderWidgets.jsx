import React from 'react';
import { FiPlus } from 'react-icons/fi';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';

const PageHeaderWidgets = ({ addNewLink, name }) => {
    return (
        <div className="d-flex align-items-center gap-2 page-header-right-items-wrapper">
            {/* Dynamic "Add New" button */}
            <Link to={addNewLink} className="btn btn-primary">
                <FiPlus size={16} className="me-2" />
                <span>{name.charAt(0).toUpperCase() + name.slice(1)}</span>
            </Link>
        </div>
    );
};

PageHeaderWidgets.propTypes = {
    addNewLink: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired
};

export default PageHeaderWidgets;
