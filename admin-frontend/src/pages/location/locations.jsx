import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import PageHeaderWidgets from '@/components/shared/pageHeader/PageHeaderWidgets';
import Footer from '@/components/shared/Footer';
import ProductTable from '@/components/location/LocationTable';

const Locations = () => {
    return (
        <>
            <PageHeader>
                <PageHeaderWidgets addNewLink="/admin/product/create" name="Add New Location" />
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <ProductTable title="Location List" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Locations;
